import torch
import torch.nn as nn


class CNNDownBlock(nn.Module):
    def __init__(self, in_size, out_size, k_size, stride=1, pad=0, bias=True):
        super(CNNDownBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(in_size, out_size, k_size, stride, pad, bias=bias),
            nn.BatchNorm2d(out_size, eps=0.0001),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)


class CNNUpBlock(nn.Module):
    def __init__(self, in_size, out_size, k_size, stride=1, pad=0, bias=True):
        super(CNNUpBlock, self).__init__()
        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, k_size, stride, pad, bias=bias),
            nn.BatchNorm2d(out_size, eps=0.0001),
        )

    def forward(self, x):
        return self.network(x)
