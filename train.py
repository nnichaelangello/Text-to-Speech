# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.shared_cnn_backbone import SharedCNNBackbone
from model.tts_head import AutoregressiveTTSHead
from model.hifigan_generator import HiFiGANGenerator
from model.discriminators import MultiPeriodDiscriminator, MultiScaleDiscriminator
from dataset import TTSDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = SharedCNNBackbone().to(device)
tts_head = AutoregressiveTTSHead().to(device)
generator = HiFiGANGenerator().to(device)
mpd = MultiPeriodDiscriminator().to(device)
msd = MultiScaleDiscriminator().to(device)

opt_backbone = optim.AdamW(backbone.parameters(), lr=config.lr_backbone)
opt_tts = optim.AdamW(tts_head.parameters(), lr=config.lr_tts_head)
opt_g = optim.AdamW(generator.parameters(), lr=config.lr_vocoder)
opt_d = optim.AdamW(list(mpd.parameters()) + list(msd.parameters()), lr=config.lr_vocoder)

dataset = TTSDataset("metadata.txt")
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                    collate_fn=dataset.collate_fn, num_workers=8, pin_memory=True)

scaler = GradScaler()

l1_loss = nn.L1Loss()
mse_loss = nn.MSELoss()

def feature_matching_loss(real_feats, fake_feats):
    loss = 0
    for real, fake in zip(real_feats, fake_feats):
        for r, f in zip(real, fake):
            loss += l1_loss(r, f)
    return loss

for epoch in range(1, config.epochs+1):

    backbone.train()
    tts_head.train()
    generator.train()
    mpd.train()
    msd.train()

    total_mel_loss = 0
    total_g_loss = 0
    total_d_loss = 0

    for phonemes, plen, feat119, mel, wav in tqdm(loader, desc=f"Epoch {epoch}"):

        feat119 = feat119.to(device)
        mel = mel.to(device)
        wav = wav.to(device)  # waveform GT

        shared = backbone(feat119)
        pred_mel = tts_head(shared)

        with autocast():
            mel_loss = l1_loss(pred_mel, mel)

        scaler.scale(mel_loss).backward()
        scaler.unscale_(opt_backbone)
        scaler.unscale_(opt_tts)
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(tts_head.parameters(), 1.0)

        scaler.step(opt_backbone)
        scaler.step(opt_tts)
        scaler.update()
        opt_backbone.zero_grad()
        opt_tts.zero_grad()

        total_mel_loss += mel_loss.item()

        with torch.no_grad():
            fake_wav = generator(pred_mel.transpose(1, 2))

        # MPD
        real_mpd = mpd(wav)
        fake_mpd = mpd(fake_wav.detach())
        d_loss_mpd = 0
        for r, f in zip(real_mpd, fake_mpd):
            d_loss_mpd += mse_loss(r, torch.ones_like(r)) + \
                           mse_loss(f, torch.zeros_like(f))

        # MSD
        real_msd = msd(wav)
        fake_msd = msd(fake_wav.detach())
        d_loss_msd = 0
        for r, f in zip(real_msd, fake_msd):
            d_loss_msd += mse_loss(r, torch.ones_like(r)) + \
                           mse_loss(f, torch.zeros_like(f))

        d_loss = d_loss_mpd + d_loss_msd

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        total_d_loss += d_loss.item()

        fake_wav = generator(pred_mel.transpose(1,2))

        # adversarial loss
        fake_mpd_2 = mpd(fake_wav)
        fake_msd_2 = msd(fake_wav)

        adv_loss = 0
        for f in fake_mpd_2:
            adv_loss += mse_loss(f, torch.ones_like(f))
        for f in fake_msd_2:
            adv_loss += mse_loss(f, torch.ones_like(f))

        # feature matching
        real_mpd_feats = mpd(wav)
        real_msd_feats = msd(wav)
        fm_loss = feature_matching_loss(real_mpd_feats, fake_mpd_2) + \
                  feature_matching_loss(real_msd_feats, fake_msd_2)

        g_loss = adv_loss + config.fm_lambda * fm_loss

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        total_g_loss += g_loss.item()

    print(f"Epoch {epoch} - MEL: {total_mel_loss/len(loader):.4f}   "
          f"GEN: {total_g_loss/len(loader):.4f}   "
          f"DISC: {total_d_loss/len(loader):.4f}")

    if epoch % 50 == 0:
        torch.save(backbone.state_dict(), f"{config.checkpoint_dir}/backbone_{epoch}.pt")
        torch.save(tts_head.state_dict(), f"{config.checkpoint_dir}/tts_head_{epoch}.pt")
        torch.save(generator.state_dict(), f"{config.checkpoint_dir}/hifigan_{epoch}.pt")
        torch.save(mpd.state_dict(), f"{config.checkpoint_dir}/mpd_{epoch}.pt")
        torch.save(msd.state_dict(), f"{config.checkpoint_dir}/msd_{epoch}.pt")
