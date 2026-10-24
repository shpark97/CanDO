"""Frozen PointNet++ S3DIS semantic-seg black-box (yanx27 pretrained pointnet2_sem_seg).

Released yanx27 checkpoint: 13 classes, 9-channel input, mIoU 0.526. Weight-frozen, eval-mode, query-counted.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_PN2_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "third_party", "Pointnet_Pointnet2_pytorch"
)
_DEFAULT_CKPT = os.path.join(
    _PN2_ROOT, "log", "sem_seg", "pointnet2_sem_seg", "checkpoints", "best_model.pth"
)
NUM_CLASSES = 13


class FrozenSemSeg(nn.Module):
    def __init__(self, ckpt: str = _DEFAULT_CKPT, device: str = "cuda"):
        super().__init__()
        if _PN2_ROOT not in sys.path:
            sys.path.insert(0, os.path.abspath(_PN2_ROOT))
        from models.pointnet2_sem_seg import get_model  # noqa: E402

        self.net = get_model(NUM_CLASSES)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        self.net.load_state_dict(state)
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.query_count = 0

    @torch.no_grad()
    def forward(self, points_9cn: torch.Tensor) -> torch.Tensor:
        """points_9cn: (B, 9, N) -> log-prob logits (B, N, 13)."""
        points_9cn = points_9cn.to(self.device)
        logits, _ = self.net(points_9cn)  # (B, N, 13) log_softmax
        self.query_count += points_9cn.shape[0]
        return logits

    def reset_queries(self):
        self.query_count = 0
