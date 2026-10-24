"""Frozen PointNet++ part-seg black-box (yanx27 pretrained pointnet2_part_seg_msg).

Uses the released yanx27 checkpoint (normal_channel=True, class mIoU 0.826), wrapped as a weight-frozen,
eval-mode black box that can only be queried. A query counter tracks the number of forward passes.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_PN2_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "third_party", "Pointnet_Pointnet2_pytorch"
)
_DEFAULT_CKPT = os.path.join(
    _PN2_ROOT, "log", "part_seg", "pointnet2_part_seg_msg", "checkpoints", "best_model.pth"
)
NUM_PARTS = 50
NUM_CATEGORIES = 16


class FrozenPartSeg(nn.Module):
    def __init__(self, ckpt: str = _DEFAULT_CKPT, normal_channel: bool = True, device: str = "cuda"):
        super().__init__()
        if _PN2_ROOT not in sys.path:
            sys.path.insert(0, os.path.abspath(_PN2_ROOT))
        from models.pointnet2_part_seg_msg import get_model  # noqa: E402

        self.normal_channel = normal_channel
        self.net = get_model(NUM_PARTS, normal_channel=normal_channel)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        self.net.load_state_dict(state)
        self.net.eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device
        self.query_count = 0

    @torch.no_grad()
    def forward(self, points_bcn: torch.Tensor, cls_idx: torch.Tensor) -> torch.Tensor:
        """points_bcn: (B, C, N) with C=6 (xyz+normal) if normal_channel else 3.
        cls_idx: (B,) category index 0-15.  Returns log-prob logits (B, N, 50)."""
        points_bcn = points_bcn.to(self.device)
        onehot = F.one_hot(cls_idx.to(self.device), NUM_CATEGORIES).float()  # (B, 16)
        logits, _ = self.net(points_bcn, onehot)  # (B, N, 50) log_softmax
        self.query_count += points_bcn.shape[0]
        return logits

    def reset_queries(self):
        self.query_count = 0


def make_input(points: torch.Tensor, normals: torch.Tensor | None, normal_channel: bool) -> torch.Tensor:
    """(N,3)[,normals (N,3)] -> (1, C, N) batched channels-first input for the frozen net."""
    if normal_channel:
        assert normals is not None, "normal_channel=True needs normals"
        feat = torch.cat([points, normals], dim=1)  # (N, 6)
    else:
        feat = points
    return feat.t().unsqueeze(0).contiguous()  # (1, C, N)


@torch.no_grad()
def batched_predict(bb: "FrozenPartSeg", kept, cls_list, min_points: int = 128):
    """One forward for many shapes. `kept`: list of (xyz_k (k,3), nrm_k (k,3)) kept-point sets.
    Pads each to a common length (repeat sampled points to clear the N>=128 floor), stacks to
    (B, 6, Kmax), runs a single batched forward, returns list of per-shape logits[:k] (k, 50)."""
    ks = [x.shape[0] for x, _ in kept]
    dev = kept[0][0].device
    kmax = max(max(ks), min_points)
    feats = []
    for (xyz_k, nrm_k), k in zip(kept, ks):
        if k < kmax:
            pad = torch.randint(k, (kmax - k,), device=dev)
            xyz_k = torch.cat([xyz_k, xyz_k[pad]])
            nrm_k = torch.cat([nrm_k, nrm_k[pad]])
        feats.append(torch.cat([xyz_k, nrm_k], dim=1).t())   # (6, kmax)
    batch = torch.stack(feats, 0).contiguous()               # (B, 6, kmax)
    logits = bb(batch, torch.tensor(cls_list, device=dev))   # (B, kmax, 50)
    return [logits[i, : ks[i]] for i in range(len(kept))]
