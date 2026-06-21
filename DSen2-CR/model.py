import torch
from torch import nn


class ResScale(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return

class ResBlock(nn.Module):
    def __init__(self, hidden_dim, out_dim, kernel_size, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size, stride, padding)

    def forward(self):
        return


class DSen2CR(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return
    