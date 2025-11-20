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
loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=dataset.collate_fn, num_workers=8, pin_memory=True)

scaler = GradScaler()

for epoch in range(1, config.epochs+1):
    backbone.train()
    tts_head.train()
    generator.train()
    mpd.train()
    msd.train()
    
    total_loss = 0.0
    
    for phonemes, plen, feat119, mel in tqdm(loader, desc=f"Epoch {epoch}"):
        feat119 = feat119.to(device)
        mel = mel.to(device)
        
        shared = backbone(feat119)
        pred_mel = tts_head(shared)
        
        with autocast():
            mel_loss = nn.L1Loss()(pred_mel, mel)
            loss = mel_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt_backbone)
        scaler.unscale_(opt_tts)
        torch.nn.utils.clip_grad_norm_(backbone.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(tts_head.parameters(), 1.0)
        scaler.step(opt_backbone)
        scaler.step(opt_tts)
        scaler.update()
        opt_backbone.zero_grad()
        opt_tts.zero_grad()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch} - Mel Loss: {total_loss/len(loader):.6f}")
    
    if epoch % 50 == 0:
        torch.save(backbone.state_dict(), f"{config.checkpoint_dir}/backbone_{epoch}.pt")
        torch.save(tts_head.state_dict(), f"{config.checkpoint_dir}/tts_head_{epoch}.pt")
        torch.save(generator.state_dict(), f"{config.checkpoint_dir}/hifigan_{epoch}.pt")