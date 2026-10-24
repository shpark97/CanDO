"""Cross-architecture test of the augmentation-gap headroom principle.

Two frozen S3DIS sem-seg models with DIFFERENT training augmentation:
  - PointNet++ (yanx27): ONLY rotate_point_cloud_z (yaw). No scale/color/jitter.
  - PointNeXt-S:         yaw + PointCloudScaling[0.9,1.1] + Jitter + Chromatic(color) transforms.

Principle: a model is sensitive to a transform (=> canonicalization headroom) IFF it was NOT
train-augmented for it. Predictions on the SAME blocks:
  - tilt (pitch):  NEITHER augments it  -> BOTH collapse        (transform-level)
  - scale:         only PointNeXt augments-> PointNet++ collapses, PointNeXt robust
  - color-scale:   only PointNeXt augments-> PointNet++ collapses, PointNeXt robust
If confirmed, headroom is the MODEL's augmentation gap, not a property of the transform.

Run:  python -m src.eval.cross_arch_auggap --n_blocks 16
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.s3dis import S3DISRooms, sample_block
from src.downstream.pointnet2_semseg import FrozenSemSeg
from src.downstream.pointnext_semseg import FrozenPointNeXt
from src.eval.canon_s3dis import block_miou
from src.eval.canon_s3dis_rooms import build_input as pn2_input
from src.samplers.classic import fps_sample
from src.zoo.canonicalize import Rso3


def transform(coord, color, c, kind, p):
    """Geometric/photometric transform about center c. Returns (coord2, color2)."""
    coord2, color2 = coord.clone(), color.clone()
    if kind == "tilt":          # pitch about x-axis through center
        coord2 = c + (coord - c) @ Rso3(torch.tensor([p, 0.0, 0.0], device=coord.device, dtype=torch.float32)).T
    elif kind == "scale":
        coord2 = c + (coord - c) * p
    elif kind == "color":
        color2 = (color * p).clamp(0, 255)
    return coord2, color2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/s3dis"); ap.add_argument("--n_blocks", type=int, default=16)
    ap.add_argument("--npoints", type=int, default=4096); ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ds = S3DISRooms(args.root, split="test", test_area=5)
    pn2 = FrozenSemSeg(); pnx = FrozenPointNeXt()
    g = torch.Generator(device="cuda"); g.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    # (transform, param) sweeps; param=identity first
    sweeps = {
        "tilt":  [0.0, np.deg2rad(20), np.deg2rad(40)],
        "scale": [1.0, 0.65, 1.5],
        "color": [1.0, 0.5, 1.6],
    }
    labels = {"tilt": ["0", "20deg", "40deg"], "scale": ["1.0", "0.65", "1.5"], "color": ["1.0", "0.5", "1.6"]}
    # acc[kind][model] = list over severities of list over blocks
    acc = {k: {"pn2": [[] for _ in sweeps[k]], "pnx": [[] for _ in sweeps[k]]} for k in sweeps}

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

        def pn2_miou(cc, ccol):
            lg = pn2(pn2_input(cc, ccol, c, rmax)).squeeze(0)
            return block_miou(lg.argmax(-1), gt)

        def pnx_miou(cc, ccol):
            lg = pnx(cc.unsqueeze(0), ccol.unsqueeze(0), cc[:, 2:3].unsqueeze(0)).squeeze(0)
            return block_miou(lg.argmax(-1), gt)

        for kind, params in sweeps.items():
            for j, p in enumerate(params):
                cc, ccol = transform(coord0, color0, c, kind, p)
                acc[kind]["pn2"][j].append(pn2_miou(cc, ccol))
                acc[kind]["pnx"][j].append(pnx_miou(cc, ccol))

    print(f"[Cross-architecture augmentation-gap @ S3DIS, n={nb} blocks]")
    print(f"  PointNet++ aug: yaw only        |  PointNeXt-S aug: yaw + scale + color + jitter\n")
    aug = {"tilt": "neither", "scale": "PointNeXt only", "color": "PointNeXt only"}
    for kind in sweeps:
        print(f"  == {kind}  (augmented by: {aug[kind]}) ==")
        m2 = [np.mean(v) for v in acc[kind]["pn2"]]
        mx = [np.mean(v) for v in acc[kind]["pnx"]]
        cols = "  ".join(f"{l:>8s}" for l in labels[kind])
        print(f"     severity:        {cols}")
        print(f"     PointNet++ mIoU: " + "  ".join(f"{v:8.3f}" for v in m2) + f"   (drop {m2[0]-min(m2):.3f})")
        print(f"     PointNeXt  mIoU: " + "  ".join(f"{v:8.3f}" for v in mx) + f"   (drop {mx[0]-min(mx):.3f})")
        print()


if __name__ == "__main__":
    main()
