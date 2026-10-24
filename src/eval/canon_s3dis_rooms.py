"""Test the augmentation-gap headroom principle on S3DIS across transforms the frozen sem-seg
model was NOT trained-augmented for.

The frozen yanx27 sem-seg training loop applies ONLY rotate_point_cloud_z (train_semseg.py:184) --
no scale, no z-shift, no color aug (their provider.py has them but they're unused). Principle:
canonicalization has headroom iff the model is NOT augmented for the transform. So scale / z-offset /
color-scale should all be SENSITIVE (room) and label-free gradient-free recoverable; yaw is the
control (augmented -> robust -> no room).

Each transform is a scalar; recovery = coarse 1-D scan + SPSA on the label-free confidence objective.
We report clean, corrupted, label-free conf-recovery, GT-oracle, and corr(confidence, mIoU).

Run:  python -m src.eval.canon_s3dis_rooms --n_blocks 20
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.s3dis import S3DISRooms, sample_block
from src.downstream.pointnet2_semseg import FrozenSemSeg
from src.eval.canon_s3dis import block_miou
from src.samplers.classic import fps_sample
from src.zoo.estimator import SPSA

# transform: (identity param, corruption sampler range, recovery scan grid)
TRANSFORMS = {
    "yaw":        {"id": 0.0, "corrupt": (0.5, 2.5),  "scan": (0.0, 6.28, 16)},   # control (augmented)
    "scale":      {"id": 1.0, "corrupt": (0.6, 1.6),  "scan": (0.5, 2.0, 16)},
    "z-offset":   {"id": 0.0, "corrupt": (-1.2, 1.2), "scan": (-2.0, 2.0, 17)},
    "color-scale":{"id": 1.0, "corrupt": (0.4, 1.7),  "scan": (0.3, 2.0, 16)},
}


def transform_coords(coord, color, c, kind, p):
    """Apply scalar transform p to (coord,color) about center c. Returns (coord2, color2)."""
    coord2, color2 = coord.clone(), color.clone()
    if kind == "yaw":
        ca, sa = np.cos(p), np.sin(p)
        dx, dy = coord[:, 0] - c[0], coord[:, 1] - c[1]
        coord2[:, 0] = c[0] + ca * dx - sa * dy
        coord2[:, 1] = c[1] + sa * dx + ca * dy
    elif kind == "scale":
        coord2 = c + (coord - c) * p
    elif kind == "z-offset":
        coord2[:, 2] = coord[:, 2] + p
    elif kind == "color-scale":
        color2 = (color * p).clamp(0, 255)
    return coord2, color2


def build_input(coord, color, c, rmax):
    f = torch.empty((coord.shape[0], 9), device=coord.device)
    f[:, 0] = coord[:, 0] - c[0]; f[:, 1] = coord[:, 1] - c[1]; f[:, 2] = coord[:, 2]
    f[:, 3:6] = color / 255.0
    f[:, 6] = coord[:, 0] / (rmax[0] + 1e-8)
    f[:, 7] = coord[:, 1] / (rmax[1] + 1e-8)
    f[:, 8] = coord[:, 2] / (rmax[2] + 1e-8)
    return f.t().unsqueeze(0).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/s3dis"); ap.add_argument("--n_blocks", type=int, default=20)
    ap.add_argument("--npoints", type=int, default=4096); ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ds = S3DISRooms(args.root, split="test", test_area=5); bb = FrozenSemSeg()
    g = torch.Generator(device="cuda"); g.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    agg = {k: {"clean": [], "corrupt": [], "conf": [], "oracle": [], "pairs": []} for k in TRANSFORMS}
    nb, tries = 0, 0
    while nb < args.n_blocks and tries < args.n_blocks * 12:
        tries += 1
        room = ds.load(int(rng.integers(len(ds))))
        block = sample_block(room, 1.0, args.npoints, generator=g)
        if block is None:
            continue
        sel = fps_sample(block["coord"], args.npoints); gt = block["seg"][sel]
        if torch.unique(gt).numel() < 2:
            continue
        nb += 1
        coord0, color0 = block["coord"][sel], block["color"][sel]
        c = block["center"].clone(); c[2] = coord0[:, 2].mean(); rmax = block["room_max"]

        def eval_at(cc, ccol, gt_=gt):
            lg = bb(build_input(cc, ccol, c, rmax)).squeeze(0)
            return block_miou(lg.argmax(-1), gt_), lg.exp().max(-1).values.mean().item()

        for kind, cfg in TRANSFORMS.items():
            mi_clean, _ = eval_at(coord0, color0)
            agg[kind]["clean"].append(mi_clean)
            lo, hi = cfg["corrupt"]; p0 = rng.uniform(lo, hi)
            ccoord, ccolor = transform_coords(coord0, color0, c, kind, p0)
            mi_cor, _ = eval_at(ccoord, ccolor)
            agg[kind]["corrupt"].append(mi_cor)
            # 1-D label-free recovery: coarse scan + SPSA, applied on the corrupted block
            s_lo, s_hi, npts = cfg["scan"]; grid = np.linspace(s_lo, s_hi, npts)
            confs, mious = [], []
            for p in grid:
                rc, rcol = transform_coords(ccoord, ccolor, c, kind, p)
                mm, cf = eval_at(rc, rcol); confs.append(cf); mious.append(mm)
                agg[kind]["pairs"].append((cf, mm))
            agg[kind]["oracle"].append(max(mious))
            p_best = float(grid[int(np.argmax(confs))])
            est = SPSA(c=0.05 * (s_hi - s_lo), avg=1)
            th = torch.tensor([p_best], dtype=torch.float32)
            neg = lambda t: -torch.tensor(eval_at(*transform_coords(ccoord, ccolor, c, kind, float(t.item())))[1])  # noqa
            for t in range(12):
                th = th - (0.2 / (1 + 0.1 * t)) * est.estimate(th, neg)
            rc, rcol = transform_coords(ccoord, ccolor, c, kind, float(th.item()))
            agg[kind]["conf"].append(eval_at(rc, rcol)[0])

    print(f"[S3DIS augmentation-gap probe, n={nb} blocks, {args.npoints} pts]  (yaw = augmented control)")
    print(f"{'transform':12s} {'clean':>6s} {'corrupt':>8s} {'conf-rec':>9s} {'oracle':>7s} {'sens.':>7s} {'corr':>6s}  verdict")
    for kind in TRANSFORMS:
        a = agg[kind]; cl = np.mean(a["clean"]); co = np.mean(a["corrupt"])
        cf = np.mean(a["conf"]); orc = np.mean(a["oracle"])
        pr = np.array(a["pairs"]); corr = float(np.corrcoef(pr[:, 0], pr[:, 1])[0, 1]) if len(pr) > 2 else float("nan")
        sens = cl - co
        room = "ROOM" if (sens > 0.05 and cf - co > 0.03) else "no room"
        print(f"{kind:12s} {cl:6.3f} {co:8.3f} {cf:9.3f} {orc:7.3f} {sens:7.3f} {corr:+6.2f}  {room}")


if __name__ == "__main__":
    main()
