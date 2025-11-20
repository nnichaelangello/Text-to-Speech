# model/tts_head.py
import torch.nn as nn

class AutoregressiveTTSHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(512, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 512, batch_first=True)
        self.mel_out = nn.Linear(512, config.n_mels)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        mel = self.mel_out(x)
        return mel