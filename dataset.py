# dataset.py
import os
import torch
from torch.utils.data import Dataset
from audio_utils import load_and_normalize
from feature_extractor import extract_features
from text.g2p_id import text_to_phonemes
from text.symbols import symbol_to_id

class TTSDataset(Dataset):
    def __init__(self, metadata_path):
        self.items = []
        with open(metadata_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                wav_path = os.path.join(os.path.dirname(metadata_path), parts[0])
                text = parts[1]
                self.items.append((wav_path, text))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, text = self.items[idx]
        audio = load_and_normalize(wav_path)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        feat_119, target_mel = extract_features(audio_tensor)
        phonemes = text_to_phonemes(text)
        phoneme_ids = torch.LongTensor([symbol_to_id.get(p, symbol_to_id['UNK']) for p in phonemes])
        return phoneme_ids, feat_119, target_mel

    def collate_fn(self, batch):
        phonemes = [b[0] for b in batch]
        feats = [b[1] for b in batch]
        mels = [b[2] for b in batch]
        phoneme_lens = torch.LongTensor([len(p) for p in phonemes])
        max_text = max(phoneme_lens)
        max_feat = max([f.shape[2] for f in feats])
        phoneme_padded = torch.zeros(len(batch), max_text, dtype=torch.long)
        feat_padded = torch.zeros(len(batch), config.total_channels, max_feat, 1)
        mel_padded = torch.zeros(len(batch), max_feat, config.n_mels)
        for i in range(len(batch)):
            phoneme_padded[i,:len(phonemes[i])] = phonemes[i]
            feat_padded[i,:,:feats[i].shape[2],:] = feats[i]
            mel_padded[i,:mels[i].shape[0],:] = mels[i]
        return phoneme_padded, phoneme_lens, feat_padded, mel_padded