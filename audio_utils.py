# audio_utils.py
import librosa
import numpy as np
import soundfile as sf
import torch

def load_and_normalize(path):
    audio, orig_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio[:,0]
    audio = audio.astype(np.float32)
    if orig_sr != config.sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=config.sr)
    max_val = np.max(np.abs(audio)) + 1e-8
    audio = audio / max_val
    return audio

def save_wav(path, audio):
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, config.sr)