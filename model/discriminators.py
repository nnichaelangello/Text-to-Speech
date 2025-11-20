# model/discriminators.py
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            self._discriminator(period) for period in [2,3,5,7,11]
        ])

    def _discriminator(self, period):
        layers = []
        channels = 32
        layers += [weight_norm(nn.Conv2d(1, channels, (5,1), (3,1), padding=(2,0)))]
        for _ in range(4):
            layers += [
                weight_norm(nn.Conv2d(channels, channels*2, (5,1), (3,1), padding=(2,0))),
                nn.LeakyReLU(0.1)
            ]
            channels *= 2
        layers += [weight_norm(nn.Conv2d(channels, 1, (3,1), 1, padding=(1,0)))]
        return nn.Sequential(*layers)

    def forward(self, x):
        rets = []
        for disc in self.discriminators:
            rets.append(disc(x.unsqueeze(1)))
        return rets

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            self._discriminator(),
            self._discriminator(),
            self._discriminator()
        ])
        self.poolings = [nn.Identity(), nn.AvgPool1d(4,2,padding=2), nn.AvgPool1d(4,2,padding=2)]

    def _discriminator(self):
        layers = []
        channels = 128
        layers += [weight_norm(nn.Conv1d(1, channels, 15, 1, padding=7))]
        for _ in range(5):
            layers += [
                weight_norm(nn.Conv1d(channels, channels*2, 41, 4, padding=20, groups=4)),
                nn.LeakyReLU(0.1)
            ]
            channels *= 2
        layers += [weight_norm(nn.Conv1d(channels, 1, 5, 1, padding=2))]
        return nn.Sequential(*layers)

    def forward(self, x):
        rets = []
        for pool, disc in zip(self.poolings, self.discriminators):
            rets.append(disc(pool(x)))
        return rets