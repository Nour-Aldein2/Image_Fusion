"""Loss name: Cloud-Adaptive Regularised Loss (L_carl)"""

import torch
from torch import nn

from config import Config


class CARLLoss(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.lamb = cfg.loss_fcn.lamb
        self.return_parts = cfg.loss_fcn.return_parts

    def forward(self, pred, target, cloudy_input, csm):
        """csm: Cloud Shadow Mask"""
        if pred.shape != target.shape or pred.shape != cloudy_input.shape:
            raise ValueError(
                "pred, target, and cloudy_input must have identical shapes. "
                f"Got pred={tuple(pred.shape)}, "
                f"target={tuple(target.shape)}, "
                f"cloudy_input={tuple(cloudy_input.shape)}."
            )

        term1 = torch.multiply(csm, (pred - target))
        term2 = torch.multiply((1 - csm), (pred - cloudy_input))   # Note: The paper makes the use of cloudy_input very confusing!

        cloud_adaptive = torch.mean(torch.abs(term1 + term2))
        target_reg = self.lamb * torch.mean(torch.abs(pred - target))
        loss = cloud_adaptive + target_reg

        if self.return_parts:
            parts = {
                "cloud_adaptive": cloud_adaptive.detach(),
                "target_reg": target_reg.detach(),
                "carl": loss.detach(),
            }
            return loss, parts

        return loss

