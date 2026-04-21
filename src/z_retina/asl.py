"""Asymmetric Loss (ASL) — optimised variant used in training."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class AsymmetricLossOptimised(nn.Module):
    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 0.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        reduction: str = "mean",
        pos_weight: Optional[Tensor] = None,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.reduction = reduction
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight.float())
        else:
            self.pos_weight = None

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        xs_pos = logits
        xs_neg = -logits
        log_p_pos = -torch.nn.functional.softplus(-xs_pos)
        log_p_neg = -torch.nn.functional.softplus(xs_neg)
        p = torch.sigmoid(logits)
        if self.clip > 0:
            p_neg_shifted = (p - self.clip).clamp(min=0)
            log_p_neg = torch.log((1.0 - p_neg_shifted).clamp(min=self.eps))
            pt_neg = p_neg_shifted
        else:
            pt_neg = p
        loss_pos = targets * ((1.0 - p) ** self.gamma_pos) * log_p_pos
        loss_neg = (1.0 - targets) * (pt_neg ** self.gamma_neg) * log_p_neg
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
            loss = -(pw * loss_pos + loss_neg)
        else:
            loss = -(loss_pos + loss_neg)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
