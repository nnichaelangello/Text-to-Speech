# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

from model.shared_cnn_backbone import SharedCNNBackbone
from model.tts_head import AutoregressiveTTSHead
from model.hifigan_generator import HiFiGANGenerator
from model.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from dataset import TTSDataset
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PhonemeFrontend(nn.Module):
    """
    Simple frontend that maps phoneme ID sequences to a continuous feature
    sequence compatible with the backbone input shape (B, feat_channels, T_mel).
    Strategy:
      - embed phoneme IDs -> (B, seq_len, emb)
      - project via 1D conv -> (B, out_channels, seq_len)
      - upsample/interpolate to match mel frame length
    """
    def __init__(self,
                 vocab_size,
                 embed_dim=256,
                 out_channels=119,
                 conv_kernel=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Conv1d(embed_dim, out_channels, conv_kernel, padding=(conv_kernel-1)//2)
        self.conv_stack = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, phoneme_ids, plen, target_len):
        """
        phoneme_ids: (B, seq_len) LongTensor (padded)
        plen: (B,) lengths
        target_len: int target mel frames length to upsample to
        returns: (B, out_channels, target_len)
        """
        emb = self.embed(phoneme_ids)
        emb = emb.transpose(1, 2) 
        x = self.proj(emb)        
        x = self.conv_stack(x)    
        if x.size(2) != target_len:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return x

backbone = SharedCNNBackbone().to(device)
tts_head = AutoregressiveTTSHead().to(device)
generator = HiFiGANGenerator().to(device)
mpd = MultiPeriodDiscriminator().to(device)
msd = MultiScaleDiscriminator().to(device)

vocab_size = getattr(config, "phoneme_vocab_size", 200)
frontend_embed_dim = getattr(config, "frontend_embed_dim", 256)
frontend_out_channels = getattr(config, "frontend_out_channels", 119)

frontend = PhonemeFrontend(vocab_size=vocab_size,
                           embed_dim=frontend_embed_dim,
                           out_channels=frontend_out_channels).to(device)

opt_backbone = optim.AdamW(backbone.parameters(), lr=config.lr_backbone)
opt_tts = optim.AdamW(list(tts_head.parameters()) + list(frontend.parameters()), lr=config.lr_tts_head)
opt_g = optim.AdamW(generator.parameters(), lr=config.lr_vocoder)
opt_d = optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=config.lr_vocoder)

dataset = TTSDataset("metadata.txt")
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                    collate_fn=dataset.collate_fn, num_workers=8, pin_memory=True)

scaler = GradScaler()

l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

fm_lambda = getattr(config, "fm_lambda", 2.0) 

def feature_matching_loss(real_feats, fake_feats):
    loss = 0.0
    for real_disc, fake_disc in zip(real_feats, fake_feats):
        if isinstance(real_disc, torch.Tensor):
            real_disc = [real_disc]
        if isinstance(fake_disc, torch.Tensor):
            fake_disc = [fake_disc]
        for r, f in zip(real_disc, fake_disc):
            loss += l1_loss(r, f)
    return loss

for epoch in range(1, config.epochs + 1):
    backbone.train()
    tts_head.train()
    frontend.train()
    generator.train()
    mpd.train()
    msd.train()

    tot_mel = 0.0
    tot_g = 0.0
    tot_d = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch}"):
        phonemes, plen, *rest = batch
        if len(rest) == 2:
            feat119, mel, wav = rest[0], rest[1][:, :], None
        elif len(rest) == 3:
            feat119, mel, wav = rest
        else:
            raise ValueError("Dataset must return (phonemes, plen, feat119?, mel, wav)")

        phonemes = phonemes.to(device)
        plen = plen.to(device)
        mel = mel.to(device)
        if wav is not None:
            wav = wav.to(device)

        # (phoneme -> mel)
        target_len = mel.size(1)  
        feat_from_phoneme = frontend(phonemes, plen, target_len) 
        shared = backbone(feat_from_phoneme)
        pred_mel = tts_head(shared) 

        mel_loss = l1_loss(pred_mel, mel)

        scaler.scale(mel_loss).backward()
        scaler.unscale_(opt_backbone)
        scaler.unscale_(opt_tts)
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(list(tts_head.parameters()) + list(frontend.parameters()), 1.0)
        scaler.step(opt_backbone)
        scaler.step(opt_tts)
        scaler.update()
        opt_backbone.zero_grad()
        opt_tts.zero_grad()

        tot_mel += mel_loss.item()

        if wav is None:
            continue

        # generate fake waveform from predicted mel
        fake_wav = generator(pred_mel.transpose(1, 2))  # (B, 1, samples) or (B, samples)
        if fake_wav.dim() == 3 and fake_wav.size(1) == 1:
            fake_wav = fake_wav.squeeze(1)

        real_mpd = mpd(wav)
        fake_mpd = mpd(fake_wav.detach())

        real_msd = msd(wav)
        fake_msd = msd(fake_wav.detach())

        d_loss = 0.0
        # MPD losses
        for r, f in zip(real_mpd, fake_mpd):
            d_loss += mse_loss(r, torch.ones_like(r)) + mse_loss(f, torch.zeros_like(f))
        # MSD losses
        for r, f in zip(real_msd, fake_msd):
            d_loss += mse_loss(r, torch.ones_like(r)) + mse_loss(f, torch.zeros_like(f))

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        tot_d += d_loss.item()

        # Generator training (G) - adversarial + feature matching
        fake_wav = generator(pred_mel.transpose(1, 2))
        if fake_wav.dim() == 3 and fake_wav.size(1) == 1:
            fake_wav = fake_wav.squeeze(1)

        fake_mpd_for_g = mpd(fake_wav)
        fake_msd_for_g = msd(fake_wav)

        adv_loss = 0.0
        for f in fake_mpd_for_g:
            adv_loss += mse_loss(f, torch.ones_like(f))
        for f in fake_msd_for_g:
            adv_loss += mse_loss(f, torch.ones_like(f))

        # feature matching - use real discriminator activations as targets
        real_mpd_feats = mpd(wav)
        real_msd_feats = msd(wav)
        fm_loss = feature_matching_loss(real_mpd_feats, fake_mpd_for_g) + \
                  feature_matching_loss(real_msd_feats, fake_msd_for_g)

        g_loss = adv_loss + fm_lambda * fm_loss

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        tot_g += g_loss.item()

    n_batches = len(loader)
    print(f"Epoch {epoch}  MEL: {tot_mel / n_batches:.6f}   GEN: {tot_g / n_batches:.6f}   DISC: {tot_d / n_batches:.6f}")

    if epoch % 50 == 0 or epoch == config.epochs:
        torch.save(backbone.state_dict(), f"{config.checkpoint_dir}/backbone_{epoch}.pt")
        torch.save(tts_head.state_dict(), f"{config.checkpoint_dir}/tts_head_{epoch}.pt")
        torch.save(generator.state_dict(), f"{config.checkpoint_dir}/hifigan_{epoch}.pt")
        torch.save(mpd.state_dict(), f"{config.checkpoint_dir}/mpd_{epoch}.pt")
        torch.save(msd.state_dict(), f"{config.checkpoint_dir}/msd_{epoch}.pt")
        torch.save(frontend.state_dict(), f"{config.checkpoint_dir}/frontend_{epoch}.pt")
