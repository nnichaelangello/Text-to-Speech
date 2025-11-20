# feature_extractor.py
import torch
import torchaudio
import python_speech_features as psf
import numpy as np

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=config.sr,
    n_fft=config.n_fft,
    hop_length=config.hop_length,
    win_length=config.win_length,
    n_mels=config.n_mels,
    f_min=config.fmin,
    f_max=config.fmax,
    power=1.0
)

def extract_features(audio_tensor):
    audio_np = audio_tensor.squeeze().cpu().numpy()
    
    mel = mel_transform(audio_tensor)
    mel = torch.log(mel + 1e-9)
    mel = mel.squeeze(0).transpose(0,1)
    
    mfcc = psf.mfcc(audio_np, samplerate=config.sr, numcep=config.n_mfcc, winlen=config.win_length/config.sr, winstep=config.hop_length/config.sr)
    delta = psf.delta(mfcc, 2)
    delta_delta = psf.delta(delta, 2)
    mfcc_full = np.concatenate([mfcc, delta, delta_delta], axis=1)
    mfcc_torch = torch.from_numpy(mfcc_full)
    
    min_T = min(mel.shape[0], mfcc_torch.shape[0])
    mel = mel[:min_T]
    mfcc_torch = mfcc_torch[:min_T]
    
    features = torch.cat([mel, mfcc_torch], dim=1)
    features = features.unsqueeze(0).unsqueeze(-1)
    return features, mel.squeeze(0)