# model/hifigan_generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

upsample_rates = [8,8,2,2]
upsample_kernel_sizes = [16,16,4,4]
upsample_initial_channel = 512
resblock_kernel_sizes = [3,7,11]
resblock_dilation_sizes = [[1,3,5],[1,3,5],[1,3,5]]

class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=d*(kernel_size-1)//2))
            for d in dilation
        ])
        self.convs1x1 = nn.ModuleList([weight_norm(nn.Conv1d(channels, channels, 1)) for _ in range(len(dilation))])

    def forward(self, x):
        for c, c1x1 in zip(self.convs, self.convs1x1):
            xt = F.leaky_relu(x, 0.1)
            xt = c(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c1x1(xt)
            x = x + xt
        return x / len(self.convs)

class HiFiGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_pre = weight_norm(nn.Conv1d(80, upsample_initial_channel, 7, 1, padding=3))
        self.ups = nn.ModuleList()
        channels = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(channels, channels//2, k, u, padding=(k-u)//2)))
            channels //= 2
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2**(i+1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

    def forward(self, x):
        x = self.conv_pre(x)
        for up in self.ups:
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            xs = 0
            for block in self.resblocks:
                xs += block(x)
            x = xs / len(resblock_kernel_sizes)
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x