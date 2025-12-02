# inference.py 
import torch
import numpy as np
from model.shared_cnn_backbone import SharedCNNBackbone
from model.tts_head import AutoregressiveTTSHead
from model.hifigan_generator import HiFiGANGenerator
from train import PhonemeFrontend   
from feature_extractor import extract_features
from text.g2p_id import text_to_phonemes
from text.symbols import symbol_to_id
from audio_utils import save_wav
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = SharedCNNBackbone().to(device)
tts_head = AutoregressiveTTSHead().to(device)
vocoder = HiFiGANGenerator().to(device)

vocab_size = getattr(config, "phoneme_vocab_size", 200)
frontend_embed_dim = getattr(config, "frontend_embed_dim", 256)
frontend_out_channels = getattr(config, "frontend_out_channels", 119)
frontend = PhonemeFrontend(vocab_size=vocab_size,
                           embed_dim=frontend_embed_dim,
                           out_channels=frontend_out_channels).to(device)

backbone.load_state_dict(torch.load("checkpoints_tts/backbone_1000.pt", map_location=device))
tts_head.load_state_dict(torch.load("checkpoints_tts/tts_head_1000.pt", map_location=device))
vocoder.load_state_dict(torch.load("checkpoints_tts/hifigan_1000.pt", map_location=device))
try:
    frontend.load_state_dict(torch.load("checkpoints_tts/frontend_1000.pt", map_location=device))
except Exception:
    pass

backbone.eval()
tts_head.eval()
vocoder.eval()
frontend.eval()

def synthesize(text, out_path="output_tts.wav"):
    phonemes = text_to_phonemes(text)
    ids = [symbol_to_id.get(p, symbol_to_id.get('UNK', 1)) for p in phonemes]
    phoneme_tensor = torch.LongTensor([ids]).to(device) 
    plen = torch.LongTensor([len(ids)]).to(device)

    frames_per_phoneme = getattr(config, "frames_per_phoneme", 12)
    target_len = max(8, int(len(ids) * frames_per_phoneme))

    with torch.no_grad():
        feat119 = frontend(phoneme_tensor, plen, target_len).to(device)  
        shared = backbone(feat119)
        mel_pred = tts_head(shared)  
        if mel_pred.dim() == 3 and mel_pred.size(1) != config.n_mels:
            mel_pred = mel_pred.transpose(1, 2)
        wav = vocoder(mel_pred).squeeze().cpu().numpy()

    wav = wav.astype(np.float32)
    maxv = np.max(np.abs(wav))
    if maxv > 0:
        wav = wav / maxv * 0.99

    save_wav(out_path, wav, sr=config.sr)
    return out_path
