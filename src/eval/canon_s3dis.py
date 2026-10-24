"""Label-free yaw canonicalization at scene scale (S3DIS), a helper for the S3DIS experiments.

The frozen sem-seg model is trained on axis-aligned rooms (walls along x/y, floor at low z).
A robot entering a room at an unknown heading sees the scene under a yaw (z-)rotation -> the
model's "walls are axis-aligned" prior breaks. We recover the heading by gradient-free
optimization of the yaw angle against a BLACK-BOX, LABEL-FREE objective (mean max-softmax /
confidence) -- no GT, no model gradients -- with the same coarse-scan + SPSA solver as on ShapeNet.

Run:  python -m src.eval.canon_s3dis --n_blocks 40 --npoints 4096
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.s3dis import NUM_CLASSES, S3DISRooms, sample_block
from src.downstream.pointnet2_semseg import FrozenSemSeg
from src.samplers.classic import fps_sample
from src.zoo.estimator import SPSA


def yaw_input(block: dict, sel: torch.Tensor, angle: float) -> torch.Tensor:
    """Build (1,9,K) sem-seg input with the block yaw-rotated by `angle` about its center (xy)."""
    coord, color = block["coord"][sel], block["color"][sel]
    center, rmax = block["center"], block["room_max"]
    ca, sa = np.cos(angle), np.sin(angle)
    dx = coord[:, 0] - center[0]
    dy = coord[:, 1] - center[1]
    rx = ca * dx - sa * dy           # block-relative, rotated
    ry = sa * dx + ca * dy
    ax = rx + center[0]              # absolute (for room-normalized feats)
    ay = ry + center[1]
    feat = torch.empty((coord.shape[0], 9), device=coord.device)
    feat[:, 0] = rx
    feat[:, 1] = ry
    feat[:, 2] = coord[:, 2]
    feat[:, 3:6] = color / 255.0
    feat[:, 6] = ax / (rmax[0] + 1e-8)
    feat[:, 7] = ay / (rmax[1] + 1e-8)
    feat[:, 8] = coord[:, 2] / (rmax[2] + 1e-8)
    return feat.t().unsqueeze(0).contiguous()


def block_miou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Mean IoU over the classes present in GT (standard S3DIS block-level mIoU)."""
    present = torch.unique(gt)
    pred_oh = pred.unsqueeze(1) == present
    gt_oh = gt.unsqueeze(1) == present
    inter = (pred_oh & gt_oh).sum(0).float()
    union = (pred_oh | gt_oh).sum(0).float()
    iou = torch.where(union > 0, inter / union, torch.ones_like(union))
    return iou.mean().item()


def confidence(bb, block, sel, angle: float) -> float:
    lg = bb(yaw_input(block, sel, angle)).squeeze(0)        # (K,13) log-prob
    return lg.exp().max(-1).values.mean().item()


def miou_at(bb, block, sel, gt, angle: float) -> float:
    lg = bb(yaw_input(block, sel, angle)).squeeze(0)
    return block_miou(lg.argmax(-1), gt)


def zoo_canonicalize_yaw(bb, block, sel, coarse=16, steps=15, c=0.1, lr=0.25, seed=0):
    """Coarse global scan over yaw (1-D, periodic, multi-modal) then SPSA refine. Label-free."""
    angs = np.linspace(0, 2 * np.pi, coarse, endpoint=False)
    confs = [confidence(bb, block, sel, float(a)) for a in angs]
    theta = torch.tensor([angs[int(np.argmax(confs))]], device=bb.device, dtype=torch.float32)
    n_eval = coarse
    neg = lambda th: -torch.tensor(confidence(bb, block, sel, float(th.item())))  # noqa: E731
    est = SPSA(c=c, avg=1)
    for t in range(steps):
        g = est.estimate(theta, neg)
        theta = theta - (lr / (1 + 0.1 * t)) * g
        n_eval += est.last_n_evals
    return float(theta.item()), n_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/s3dis")
    ap.add_argument("--n_blocks", type=int, default=40)
    ap.add_argument("--npoints", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ds = S3DISRooms(args.root, split="test", test_area=5)
    bb = FrozenSemSeg()
    g = torch.Generator(device="cuda"); g.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    grid = np.linspace(0, 2 * np.pi, 16, endpoint=False)

    clean, corrupt, zoo, oracle, n_evals = [], [], [], [], []
    nb, tries = 0, 0
    while nb < args.n_blocks and tries < args.n_blocks * 10:
        tries += 1
        room = ds.load(int(rng.integers(len(ds))))
        block = sample_block(room, block_size=1.0, min_points=args.npoints, generator=g)
        if block is None:
            continue
        sel = fps_sample(block["coord"], args.npoints)
        gt = block["seg"][sel]
        if torch.unique(gt).numel() < 2:                    # skip trivial single-class blocks
            continue
        nb += 1
        clean.append(miou_at(bb, block, sel, gt, 0.0))
        a0 = rng.uniform(0, 2 * np.pi)                       # misorientation (unknown heading)
        corrupt.append(miou_at(bb, block, sel, gt, a0))
        # recover the residual rotation to undo (relative to the corrupted frame a0)
        d, ne = zoo_canonicalize_yaw(bb, block, sel, seed=nb)
        zoo.append(miou_at(bb, block, sel, gt, a0 + d)); n_evals.append(ne)
        oracle.append(max(miou_at(bb, block, sel, gt, a0 + a) for a in grid))

    print(f"[yaw canonicalization @ S3DIS Area_5, n={nb} blocks, {args.npoints} pts]")
    print(f"  clean (aligned)        mIoU = {np.mean(clean):.4f}")
    print(f"  corrupted (yaw)        mIoU = {np.mean(corrupt):.4f}")
    print(f"  CanDO (scan+SPSA, label-free) = {np.mean(zoo):.4f}   (avg {np.mean(n_evals):.0f} queries/block)")
    print(f"  oracle (grid-16, GT)   mIoU = {np.mean(oracle):.4f}")
    gap = np.mean(oracle) - np.mean(corrupt)
    print(f"  -> CanDO recovers {np.mean(zoo)-np.mean(corrupt):+.4f} mIoU label-free "
          f"({100*(np.mean(zoo)-np.mean(corrupt))/(gap+1e-9):.0f}% of oracle gap)")


if __name__ == "__main__":
    main()
