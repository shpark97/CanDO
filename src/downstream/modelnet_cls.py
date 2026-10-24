"""Frozen ModelNet40 classification black-box (yanx27 pointnet2_cls_ssg, no normals, 1024 pts).

yanx27 trains this classifier without rotation augmentation, so it is rotation-sensitive (the classic
ModelNet40 rotation-robustness setting). Label-free objective for
canonicalization is simply the max softmax prob (one global class). Weight-frozen, query-counted.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_PN2_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "Pointnet_Pointnet2_pytorch")
_CKPT = os.path.join(_PN2_ROOT, "log", "classification", "pointnet2_ssg_wo_normals",
                     "checkpoints", "best_model.pth")
NUM_CLASSES = 40


class FrozenModelNetCls(nn.Module):
    def __init__(self, ckpt: str = _CKPT, device: str = "cuda"):
        super().__init__()
        for p in (os.path.abspath(_PN2_ROOT), os.path.abspath(os.path.join(_PN2_ROOT, "models"))):
            if p not in sys.path:
                sys.path.insert(0, p)  # cls models do a bare `import pointnet2_utils`
        from models.pointnet2_cls_ssg import get_model

        self.net = get_model(NUM_CLASSES, normal_channel=False)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        self.net.load_state_dict(state)
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.query_count = 0

    @torch.no_grad()
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz (B,N,3) -> log-prob logits (B,40)."""
        xyz = xyz.to(self.device).transpose(1, 2).contiguous()  # (B,3,N)
        logits, _ = self.net(xyz)  # (B,40) log_softmax
        self.query_count += xyz.shape[0]
        return logits

    def reset_queries(self):
        self.query_count = 0
