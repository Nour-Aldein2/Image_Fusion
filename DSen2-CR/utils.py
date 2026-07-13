import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor

from config import Config


class CSMMask(nn.Module):
    """
    Creates the binary Cloud and Cloud-Shadow Mask (CSM) used by CARL.

    Input:
        optical: Sentinel-2 image with shape [B, 13, H, W]

    Band order:
        B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12

    Output:
        0: clear pixel
        1: cloud or cloud-shadow pixel
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.cloud_threshold = cfg.csm_mask.cloud_threshold
        self.reflectance_scale = cfg.csm_mask.reflectance_scale
        self.register_buffer("avg_kernel", torch.ones(1, 1, 7, 7) / 49.0, persistent=False)

    @torch.no_grad()
    def forward(self, optical: Tensor):
        if optical.ndim != 4 or optical.shape[1] != 13:
            raise ValueError(f"optical must have shape [B, 13, H, W]. Got {tuple(optical.shape)}.")

        optical = optical.float() / self.reflectance_scale

        cloud_mask = self._cloud_mask(optical)
        shadow_mask = self._shadow_mask(optical)

        return torch.logical_or(cloud_mask, shadow_mask).float()

    def _cloud_mask(self, optical: Tensor):
        b1 = optical[:, 0:1]
        b2 = optical[:, 1:2]
        b3 = optical[:, 2:3]
        b4 = optical[:, 3:4]
        b10 = optical[:, 10:11]
        b11 = optical[:, 11:12]

        score = torch.ones_like(b1)
        score = torch.minimum(score, self._rescale(b2, 0.10, 0.50))
        score = torch.minimum(score, self._rescale(b1, 0.10, 0.30))
        score = torch.minimum(score, self._rescale(b1 + b10, 0.15, 0.20))
        score = torch.minimum(score, self._rescale(b4 + b3 + b2, 0.20, 0.80))

        ndsi = self._normalised_difference(b3, b11)
        score = torch.minimum(score, self._rescale(ndsi, 0.80, 0.60))

        score = F.max_pool2d(F.pad(score, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
        score = -F.max_pool2d(F.pad(-score, (2, 2, 2, 2), mode="reflect"), kernel_size=5, stride=1)
        score = F.conv2d(score, self.avg_kernel.to(score), padding=3)
        score = torch.clamp(score, min=1e-5, max=1.0)

        return score >= self.cloud_threshold

    def _shadow_mask(self, optical: Tensor):
        b2 = optical[:, 1:2]
        b8 = optical[:, 7:8]
        b11 = optical[:, 11:12]

        csi = (b8 + b11) / 2.0

        csi_min = torch.amin(csi, dim=(-2, -1), keepdim=True)
        csi_mean = torch.mean(csi, dim=(-2, -1), keepdim=True)
        b2_min = torch.amin(b2, dim=(-2, -1), keepdim=True)
        b2_mean = torch.mean(b2, dim=(-2, -1), keepdim=True)

        csi_threshold = csi_min + 0.75 * (csi_mean - csi_min)
        b2_threshold = b2_min + (5.0 / 6.0) * (b2_mean - b2_min)

        shadow_mask = torch.logical_and(csi < csi_threshold, b2 < b2_threshold).float()
        shadow_mask = F.avg_pool2d(shadow_mask, kernel_size=5, stride=1, padding=2)

        return shadow_mask >= (13.0 / 25.0)

    @staticmethod
    def _rescale(x: Tensor, lower: float, upper: float):
        return (x - lower) / (upper - lower)

    @staticmethod
    def _normalised_difference(x1: Tensor, x2: Tensor):
        denominator = x1 + x2
        denominator = torch.where(denominator == 0, torch.full_like(denominator, 1e-3), denominator)
        return (x1 - x2) / denominator