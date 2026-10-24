"""Frozen PointNeXt-S ModelNet40 classifier black-box (cross-architecture cls for recovery generality).

Same role as FrozenModelNetCls (PointNet++) but a different, modern architecture, trained on the SAME
modelnet40_normal_resampled data and WITHOUT rotation augmentation, so it is pose-sensitive and an
SO(3) corruption leaves recoverable headroom. Used to show CanDO's classification recovery is not a
PointNet++ artifact. Weight-frozen, eval-mode, query-counted; forward (B,N,3) -> log_softmax (B,40).
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_PNX_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "PointNeXt")
_CFG = os.path.join(_PNX_ROOT, "cfgs", "modelnet40ply2048", "pointnext-s.yaml")
_CKPT = os.path.join(os.path.dirname(__file__), "..", "..", "pretrained", "pointnext_modelnet",
                     "best_model.pth")


def build_pointnext_cls(cfg_path: str = _CFG, device: str = "cuda"):
    """Build a PointNeXt-S BaseCls (40-way) from the modelnet config; returns an untrained nn.Module."""
    if os.path.abspath(_PNX_ROOT) not in sys.path:
        sys.path.insert(0, os.path.abspath(_PNX_ROOT))
    from openpoints.models import build_model_from_cfg
    from openpoints.utils import EasyConfig
    # pointvector is not imported by the backbone __init__, so its @MODELS.register_module()
    # never runs unless we import it explicitly. Harmless if already registered.
    try:
        import openpoints.models.backbone.pointvector  # noqa: F401
    except Exception:
        pass

    cfg = EasyConfig(); cfg.load(cfg_path, recursive=True)
    # drop the training criterion so the module is a pure forward map (B,N,3) -> (B,40) logits
    cfg.model.criterion_args = None
    net = build_model_from_cfg(cfg.model)
    return net.to(device)


class FrozenPointNeXtCls(nn.Module):
    def __init__(self, ckpt: str = _CKPT, cfg_path: str = _CFG, device: str = "cuda"):
        super().__init__()
        self.net = build_pointnext_cls(cfg_path=cfg_path, device=device)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("model_state_dict", state)
        self.net.load_state_dict(state)
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.query_count = 0

    @torch.no_grad()
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """(B,N,3) -> (B,40) log_softmax, matching FrozenModelNetCls."""
        xyz = xyz.to(self.device)
        logits = self.net(xyz.contiguous())  # BaseCls.forward(pos) -> (B,40) raw logits
        self.query_count += xyz.shape[0]
        return F.log_softmax(logits, dim=-1)

    def reset_queries(self):
        self.query_count = 0
