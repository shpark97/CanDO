"""ShapeNetPart part-segmentation metric: category-masked instance mIoU (standard protocol).

For a shape of category c with global part set P_c:
  iou_p = |pred==p & gt==p| / |pred==p | gt==p|   (=1 if the union is empty)
  shape_mIoU = mean over p in P_c
instance mIoU = mean shape_mIoU over all shapes.
Predictions are argmaxed over P_c only (the category is known at eval time).
"""
from __future__ import annotations

import torch

from src.data.shapenet_part import CATEGORIES

# cls id -> list of global part ids
PART_RANGES: dict[int, list[int]] = {
    c: list(range(lo, hi + 1)) for _, _, c, (lo, hi) in CATEGORIES
}


def mask_argmax(logits: torch.Tensor, cls_id: int) -> torch.Tensor:
    """argmax over only the parts belonging to category cls_id. logits (N, 50) -> (N,)."""
    parts = PART_RANGES[cls_id]
    masked = torch.full_like(logits, float("-inf"))
    masked[:, parts] = logits[:, parts]
    return masked.argmax(dim=1)


_PARTS_TENSOR: dict[tuple[int, torch.device], torch.Tensor] = {}


def _parts(cls_id: int, device: torch.device) -> torch.Tensor:
    key = (cls_id, device)
    if key not in _PARTS_TENSOR:
        _PARTS_TENSOR[key] = torch.tensor(PART_RANGES[cls_id], device=device)
    return _PARTS_TENSOR[key]


def shape_miou(pred: torch.Tensor, gt: torch.Tensor, cls_id: int) -> float:
    """Mean IoU over the category's parts for a single shape (vectorized, single GPU sync)."""
    parts = _parts(cls_id, pred.device)              # (Pc,)
    pred_oh = pred.unsqueeze(1) == parts             # (N, Pc)
    gt_oh = gt.unsqueeze(1) == parts
    inter = (pred_oh & gt_oh).sum(0).float()         # (Pc,)
    union = (pred_oh | gt_oh).sum(0).float()
    iou = torch.where(union > 0, inter / union, torch.ones_like(union))  # absent-in-both -> 1
    return iou.mean().item()                          # one sync


def instance_miou(shape_mious: list[float]) -> float:
    return sum(shape_mious) / max(len(shape_mious), 1)
