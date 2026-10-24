"""Seed-robust 3-seed S3DIS tilt canonicalization on the FROZEN PointNeXt-S scene model.

Cross-architecture recovery companion to canon_s3dis_tilt_seeds.py (which runs the SAME deterministic
2-D grid recovery on PointNet++). Same corruption distribution, same 13x13 recovery grid, same 7x7
oracle grid, same seeds and n_blocks, so the PointNeXt-S row is directly comparable to the PointNet++
row in Table 1, showing that the label-free tilt recovery is architecture-independent.

PointNeXt specifics vs PointNet++: the adapter returns RAW logits (-> softmax for confidence, not exp),
and the input is (pos, color, heights) with the model re-centering internally. The tilt is applied to the
room coords about the block center (z = block mean), exactly as canon_s3dis_tilt.tilt_input rotates for
PointNet++.

Run (sequential, one process -> no GPU contention):
  PYTHONPATH=third_party/PointNeXt:. python -m src.eval.canon_s3dis_tilt_pointnext --n_blocks 16 --seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.s3dis import S3DISRooms, sample_block
from src.downstream.pointnext_semseg import FrozenPointNeXt
from src.eval.canon_s3dis import block_miou
from src.samplers.classic import fps_sample
from src.zoo.canonicalize import Rso3


def horiz(v2):
    """2-vector (a,b) -> axis-angle 3-vector (a,b,0): tilt about a horizontal axis (yaw is free)."""
    return torch.tensor([float(v2[0]), float(v2[1]), 0.0], device="cuda", dtype=torch.float32)


def tilt_coord(coord0, c, v):
    """Rotate room coords about the block center (z = block mean) by axis-angle v. Mirrors
    canon_s3dis_tilt.tilt_input: d = (coord-c)R^T, ac = d + c (returns the room-frame tilted coords)."""
    return c + (coord0 - c) @ Rso3(v).T


def _logits(pnx, ac, color0):
    """(N,3) tilted room coords + (N,3) color -> (N,13) raw logits, with the model re-centering internally."""
    return pnx(ac.unsqueeze(0), color0.unsqueeze(0), ac[:, 2:3].unsqueeze(0)).squeeze(0)


def confidence(pnx, coord0, color0, c, v) -> float:
    """Label-free objective: mean max-softmax confidence (PointNeXt returns raw logits, so softmax)."""
    return _logits(pnx, tilt_coord(coord0, c, v), color0).softmax(-1).max(-1).values.mean().item()


def miou(pnx, coord0, color0, c, gt, v) -> float:
    return block_miou(_logits(pnx, tilt_coord(coord0, c, v), color0).argmax(-1), gt)


def recover_tilt_deterministic(pnx, coord0, color0, c, v0, grid_n=13, span=1.0):
    """Deterministic 2-D recovery: a DENSE grid scan over the tilt-undo (no RNG). 13x13=169 queries,
    budget-matched to the 2-D random search. Returns (recovered v, n_queries)."""
    g = np.linspace(-span, span, grid_n)
    nq = 0
    best_w, best_c = np.zeros(2), -1.0
    for a in g:
        for b in g:
            cf = confidence(pnx, coord0, color0, c, v0 + horiz([a, b])); nq += 1
            if cf > best_c:
                best_c, best_w = cf, np.array([a, b])
    return v0 + horiz(best_w), nq


def run_seed(pnx, ds, seed, n_blocks, npoints, grid_n=13):
    g = torch.Generator(device="cuda"); g.manual_seed(seed); rng = np.random.default_rng(seed)
    grid = np.linspace(-0.9, 0.9, 7)
    Z = torch.zeros(3, device="cuda")
    clean, corrupt, rec, oracle, nqs = [], [], [], [], []
    nb, tries = 0, 0
    while nb < n_blocks and tries < n_blocks * 12:
        tries += 1
        room = ds.load(int(rng.integers(len(ds))))
        block = sample_block(room, 1.0, npoints, generator=g)
        if block is None:
            continue
        sel = fps_sample(block["coord"], npoints); gt = block["seg"][sel]
        if torch.unique(gt).numel() < 2:
            continue
        nb += 1
        coord0, color0 = block["coord"][sel], block["color"][sel]
        c = block["center"].clone(); c[2] = coord0[:, 2].mean()
        clean.append(miou(pnx, coord0, color0, c, gt, Z))
        ang = rng.uniform(0.3, 0.9); phi = rng.uniform(0, 2 * np.pi)
        v0 = horiz([np.cos(phi) * ang, np.sin(phi) * ang])
        corrupt.append(miou(pnx, coord0, color0, c, gt, v0))
        v, nq = recover_tilt_deterministic(pnx, coord0, color0, c, v0, grid_n=grid_n); nqs.append(nq)
        rec.append(miou(pnx, coord0, color0, c, gt, v))
        oracle.append(max(miou(pnx, coord0, color0, c, gt, v0 + horiz([a, b])) for a in grid for b in grid))
    return {"clean": np.mean(clean), "corrupt": np.mean(corrupt), "rec": np.mean(rec),
            "oracle": np.mean(oracle), "nq": np.mean(nqs), "nb": nb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/s3dis"); ap.add_argument("--n_blocks", type=int, default=16)
    ap.add_argument("--npoints", type=int, default=4096); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--grid_n", type=int, default=13, help="dense tilt grid resolution (queries = grid_n^2)")
    args = ap.parse_args()

    ds = S3DISRooms(args.root, split="test", test_area=5); pnx = FrozenPointNeXt()
    rows = []
    for s in args.seeds:
        r = run_seed(pnx, ds, s, args.n_blocks, args.npoints, grid_n=args.grid_n)
        rows.append(r)
        print(f"  seed {s} (n={r['nb']}): clean {r['clean']:.3f}  corrupt {r['corrupt']:.3f}  "
              f"recover {r['rec']:.3f}  oracle {r['oracle']:.3f}  ({r['nq']:.0f} q/block)", flush=True)

    def ms(key):
        v = np.array([r[key] for r in rows]); return v.mean(), v.std()
    print(f"\n[PointNeXt-S S3DIS tilt canonicalization, DETERMINISTIC recovery, seeds={args.seeds}, n={args.n_blocks}/seed]")
    for key, lab in [("corrupt", "corrupted (tilt)"), ("rec", "recovered (label-free)"),
                     ("oracle", "oracle (GT)"), ("clean", "clean (upright)")]:
        m, sd = ms(key); print(f"  {lab:24s} = {m:.4f} +/- {sd:.4f}")
    mc, _ = ms("corrupt"); mr, _ = ms("rec"); mo, _ = ms("oracle")
    print(f"  -> recovers {mr-mc:+.4f} mIoU label-free ({100*(mr-mc)/(mo-mc+1e-9):.0f}% of oracle gap), "
          f"std across seeds = {ms('rec')[1]:.4f}")


if __name__ == "__main__":
    main()
