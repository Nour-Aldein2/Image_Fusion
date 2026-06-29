import torch
from torch import nn
from torch import Tensor

from config import Config


class ResBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(cfg.res_block.feature_dim, cfg.res_block.feature_dim, cfg.res_block.kernel_size, cfg.res_block.stride, cfg.res_block.padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(cfg.res_block.feature_dim, cfg.res_block.feature_dim, cfg.res_block.kernel_size, cfg.res_block.stride, cfg.res_block.padding)

    def forward(self, x: Tensor):
        z = self.conv1(x)
        z = self.relu(z)
        z = self.conv2(z)
        z = z * self.cfg.res_block.res_scale_const
        return x + z


class DSen2CR(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv2d(cfg.model.input_dim, cfg.res_block.feature_dim, cfg.model.kernel_size,
                               cfg.model.stride, cfg.model.padding)
        self.relu = nn.ReLU()
        self.res_blocks = nn.Sequential(*[ResBlock(cfg) for _ in range(cfg.model.res_units_count)])
        self.conv2 = nn.Conv2d(cfg.res_block.feature_dim, cfg.model.out_dim, cfg.model.kernel_size,
                               cfg.model.stride, cfg.model.padding)

    def forward(self, sar: Tensor, optical: Tensor):
        x = torch.cat([sar, optical], dim=1)
        z = self.conv1(x)
        z = self.relu(z)
        z = self.res_blocks(z)
        z = self.conv2(z)
        return z + optical
    