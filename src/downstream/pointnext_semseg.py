"""Frozen PointNeXt-S S3DIS semantic-seg black-box.

Different architecture AND different training augmentation than yanx27 PointNet++ sem-seg:
PointNeXt-S/S3DIS trains with PointCloudScaling [0.9,1.1], PointCloudJitter, color transforms, and
yaw-only rotation (angle=[0,0,1]). Used to test the augmentation-gap headroom principle across
architectures. Weight-frozen, eval-mode, query-counted.

Input recipe (PointNeXt val pipeline, reproduced):
  pos      = block xyz, then PointCloudXYZAlign (pos-=mean; pos[:,z]-=min z)
  features = [ChromaticNormalize(color), heights]  (in_channels=4); heights = room z (above floor)
Model: cfgs/s3dis/pointnext-s.yaml (BaseSeg + PointNextEncoder, 13 classes).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_PNX_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "PointNeXt")
_CFG = os.path.join(_PNX_ROOT, "cfgs", "s3dis", "pointnext-s.yaml")
_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "pretrained", "pointnext_s3dis", "checkpoint",
    "s3dis-train-pointnext-s-ngpus1-seed1742-20220525-162639-Gz4ViiL95b6KP9MMH2ytG3_ckpt_best.pth")
NUM_CLASSES = 13
_COLOR_MEAN = [0.5136457, 0.49523646, 0.44921124]
_COLOR_STD = [0.18308958, 0.18415008, 0.19252081]


class FrozenPointNeXt(nn.Module):
    def __init__(self, cfg_path: str = _CFG, ckpt: str = _CKPT, device: str = "cuda"):
        super().__init__()
        if os.path.abspath(_PNX_ROOT) not in sys.path:
            sys.path.insert(0, os.path.abspath(_PNX_ROOT))
        from openpoints.models import build_model_from_cfg
        from openpoints.utils import EasyConfig, load_checkpoint

        cfg = EasyConfig(); cfg.load(cfg_path, recursive=True)
        self.net = build_model_from_cfg(cfg.model)
        load_checkpoint(self.net, pretrained_path=ckpt)
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.query_count = 0
        self.cmean = torch.tensor(_COLOR_MEAN, device=device)
        self.cstd = torch.tensor(_COLOR_STD, device=device)

    @torch.no_grad()
    def forward(self, pos: torch.Tensor, color: torch.Tensor, heights: torch.Tensor) -> torch.Tensor:
        """pos (B,N,3) room xyz; color (B,N,3) 0-255; heights (B,N,1) room z. -> logits (B,N,13)."""
        pos, color, heights = pos.to(self.device), color.to(self.device), heights.to(self.device)
        # PointCloudXYZAlign (per-sample): center xy, floor z to 0
        p = pos - pos.mean(dim=1, keepdim=True)
        p[..., 2] = p[..., 2] - p[..., 2].min(dim=1, keepdim=True).values
        # ChromaticNormalize
        c = color / 255.0 if color.max() > 1 else color
        c = (c - self.cmean) / self.cstd
        x = torch.cat([c, heights], dim=-1).transpose(1, 2).contiguous()  # (B,4,N)
        logits = self.net({"pos": p.contiguous(), "x": x})  # (B,13,N)
        self.query_count += pos.shape[0]
        if logits.dim() == 3 and logits.shape[1] == NUM_CLASSES:
            logits = logits.transpose(1, 2)  # -> (B,N,13)
        return logits.contiguous()

    def reset_queries(self):
        self.query_count = 0
