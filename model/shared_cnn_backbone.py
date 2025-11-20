# model/shared_cnn_backbone.py
import torch
import torch.nn as nn

class SharedCNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,3))
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2))
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2))
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, 3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2,2))
            )
        ])
        self.gap = nn.AdaptiveAvgPool2d((1,None))
        self.dropout = nn.Dropout(0.4)
        self.proj = nn.Linear(512, 512)

    def forward(self, x):
        x = x.permute(0,3,1,2)
        for block in self.conv_blocks:
            x = block(x)
        x = self.gap(x).squeeze(2)
        x = x.transpose(1,2)
        x = self.dropout(x)
        x = self.proj(x)
        return x