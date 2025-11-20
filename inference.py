# inference.py
import torch
from model.shared_cnn_backbone import SharedCNNBackbone
from model.tts_head import AutoregressiveTTSHead
from model.hifigan_generator import HiFiGANGenerator
from feature_extractor import extract_features
from text.g2p_id import text_to_phonemes
from text.symbols import symbol_to_id
from audio_utils import save_wav

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = SharedCNNBackbone().to(device)
tts_head = AutoregressiveTTSHead().to(device)
vocoder = HiFiGANGenerator().to(device)

backbone.load_state_dict(torch.load("checkpoints_tts/backbone_1000.pt", map_location=device))
tts_head.load_state_dict(torch.load("checkpoints_tts/tts_head_1000.pt", map_location=device))
vocoder.load_state_dict(torch.load("checkpoints_tts/hifigan_1000.pt", map_location=device))

backbone.eval()
tts_head.eval()
vocoder.eval()

def synthesize(text):
    phonemes = text_to_phonemes(text)
    ids = torch.LongTensor([symbol_to_id.get(p, symbol_to_id['UNK']) for p in phonemes])
    
    dummy_audio = torch.zeros(1, config.sr * 5)
    feat119, _ = extract_features(dummy_audio.unsqueeze(0))
    feat119 = feat119.to(device)
    
    with torch.no_grad():
        shared = backbone(feat119)
        mel_pred = tts_head(shared)
        mel_pred = mel_pred.transpose(1,2)
        wav = vocoder(mel_pred).squeeze().cpu().numpy()
    
    save_wav("output_tts_indonesia.wav", wav)
    print("Selesai! File: output_tts_indonesia.wav")

synthesize("Selamat datang di sistem Text-to-Speech bahasa Indonesia tercanggih tahun 2025.")