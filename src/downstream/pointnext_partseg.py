"""Frozen PointNeXt-S ShapeNetPart part-seg black-box.

Trained on ShapeNetPart with cfgs/shapenetpart/pointnext-s.yaml. Like the yanx27 PointNet++ part-seg
model it uses NO rotation augmentation (train transforms: PointsToTensor, PointCloudScaling,
PointCloudCenterAndNormalize, PointCloudJitter, ChromaticDropGPU), so it is orientation-sensitive and
exposes an azimuth / SO(3) augmentation gap. Used to show part-seg recovery is architecture-independent.

Input recipe (PointNeXt shapenetpart val pipeline, reproduced exactly):
  given RAW xyz (N,3) and normals (N,3):
    heights = xyz[:, gravity_dim=1] - min        (computed on raw coords, before centering)
    pos     = xyz - mean(xyz); pos /= max||pos||  (PointCloudCenterAndNormalize)
    x       = cat([pos, normals, heights])        (feature_keys = pos,x,heights -> in_channels 7)
    cls     = object category index (one per shape)
Weight-frozen, eval-mode, query-counted. Returns per-point log-prob over the 50 parts to match the
FrozenPartSeg interface used by the recovery code (confidence does .exp().max()).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_PNX_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "PointNeXt")
_CFG = os.path.join(_PNX_ROOT, "cfgs", "shapenetpart", "pointnext-s.yaml")
_CKPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "pretrained", "pointnext_shapenetpart", "best.pth")
NUM_PARTS = 50
GRAVITY_DIM = 1  # cfg kwargs gravity_dim: 1


class FrozenPointNeXtPartSeg(nn.Module):
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

    @torch.no_grad()
    def forward(self, points_bcn: torch.Tensor, cls_idx: torch.Tensor) -> torch.Tensor:
        """points_bcn: (B, 6, N) RAW [xyz | normal]; cls_idx: (B,) category 0-15.
        Returns log-prob (B, N, 50) over the 50 parts."""
        points_bcn = points_bcn.to(self.device)
        B = points_bcn.shape[0]
        pos = points_bcn[:, 0:3].transpose(1, 2).contiguous()      # (B,N,3) raw xyz
        normals = points_bcn[:, 3:6].transpose(1, 2).contiguous()  # (B,N,3)
        # PointCloudCenterAndNormalize (gravity_dim=1), reproduced
        height = pos[:, :, GRAVITY_DIM:GRAVITY_DIM + 1]
        heights = height - height.min(dim=1, keepdim=True).values   # (B,N,1) raw-scale
        pos = pos - pos.mean(dim=1, keepdim=True)
        m = pos.norm(dim=2, keepdim=True).max(dim=1, keepdim=True).values  # (B,1,1)
        pos = pos / m
        x = torch.cat([pos, normals, heights], dim=-1).transpose(1, 2).contiguous()  # (B,7,N)
        cls = cls_idx.to(self.device).long().view(B, 1)
        logits = self.net({"pos": pos.contiguous(), "x": x, "cls": cls})  # (B,50,N)
        self.query_count += B
        logits = logits.transpose(1, 2).contiguous()                 # (B,N,50)
        return F.log_softmax(logits, dim=-1)

    def reset_queries(self):
        self.query_count = 0
