"""
CAFormer-B36 (timm) + ML-Decoder head for multi-label fundus classification.

Matches the production setup: MetaFormer CAFormer backbone, spatial tokens fed
to ML-Decoder (Ridnik et al. decoder variant bundled under ``third_party/``).
"""

from __future__ import annotations

from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch import Tensor

from z_retina.third_party.ml_decoder import MLDecoder


def load_caformer_checkpoint(model: "CAFormerModel", ckpt_path: Path | str, device=None) -> dict:
    ckpt_path = Path(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt)
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint has no model state_dict")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        n = min(12, len(missing))
        print(f"[resume] missing keys ({len(missing)}, show {n}): {list(missing)[:n]}")
    if unexpected:
        print(f"[resume] unexpected keys: {len(unexpected)}")
    return ckpt


class CAFormerModel(nn.Module):
    """CAFormer backbone (timm) + ML-Decoder over spatial feature tokens."""

    def __init__(
        self,
        model_name: str = "caformer_b36",
        num_classes: int = 15,
        n_channels: int = 3,
        drop_path_rate: float = 0.2,
        grad_checkpointing: bool = True,
        timm_pretrained: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.n_channels = n_channels

        self.backbone = timm.create_model(
            model_name,
            pretrained=timm_pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
            in_chans=n_channels,
        )

        if grad_checkpointing:
            try:
                self.backbone.set_grad_checkpointing(True)
            except Exception:
                pass

        embed_dim = int(self.backbone.num_features)
        self.head = MLDecoder(
            num_classes=num_classes,
            initial_num_features=embed_dim,
            num_of_groups=num_classes,
            decoder_embedding=768,
            zsl=0,
        )

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(True)

    def _iter_blocks(self) -> list[nn.Module]:
        blocks: list[nn.Module] = []
        for st in self.backbone.stages:
            for blk in st.blocks:
                blocks.append(blk)
        return blocks

    def unfreeze_last_n_blocks(self, n: int = 6) -> None:
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.head.parameters():
            p.requires_grad_(True)
        blocks = self._iter_blocks()
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad_(True)

    def unfreeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x: Tensor) -> Tensor:
        feat = self.backbone.forward_features(x)
        b, c, h, w = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        return self.head(tokens)
