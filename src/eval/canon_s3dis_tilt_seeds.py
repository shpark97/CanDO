"""Seed-robust 3-seed statistics for S3DIS tilt canonicalization.

In 2-D, budget-matched random search matched/beat CMA-ES/SPSA,
and stochastic optimizers add seed variance. So the CANONICAL recovery here is DETERMINISTIC -- a 2-D
coarse grid scan + coordinate finite-difference refine (no RNG at all). That removes optimizer-seed
variance entirely; the only remaining variance is block sampling, which the 3 data seeds (0,1,2)
capture as mean +/- std. The per-input output of the optimizer is thus a deterministic function of the input.

Run (sequential, one process -> no GPU contention):
  python -m src.eval.canon_s3dis_tilt_seeds --n_blocks 16 --seeds 0 1 2
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.s3dis import S3DISRooms, sample_block
from src.downstream.pointnet2_semseg import FrozenSemSeg
from src.eval.canon_s3dis_tilt import confidence, horiz, miou
from src.samplers.classic import fps_sample


def recover_tilt_deterministic(bb, block, sel, v0, grid_n=13, span=1.0):
    """Deterministic 2-D recovery: a DENSE grid scan over the tilt-undo (no RNG, no refine).
    13x13=169 queries is budget-matched to the random search that won in 2-D (~160), so this is both
    fully seed-deterministic AND as strong as the best 2-D optimizer. Returns (recovered v, n_queries)."""
    g = np.linspace(-span, span, grid_n)
    nq = 0
    best_w, best_c = np.zeros(2), -1.0
    for a in g:
        for b in g:
            c = confidence(bb, block, sel, v0 + horiz([a, b])); nq += 1
            if c > best_c:
                best_c, best_w = c, np.array([a, b])
    return v0 + horiz(best_w), nq


def run_seed(bb, ds, seed, n_blocks, npoints, grid_n=13):
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
        clean.append(miou(bb, block, sel, gt, Z))
        ang = rng.uniform(0.3, 0.9); phi = rng.uniform(0, 2 * np.pi)
        v0 = horiz([np.cos(phi) * ang, np.sin(phi) * ang])
        corrupt.append(miou(bb, block, sel, gt, v0))
        v, nq = recover_tilt_deterministic(bb, block, sel, v0, grid_n=grid_n); nqs.append(nq)
        rec.append(miou(bb, block, sel, gt, v))
        oracle.append(max(miou(bb, block, sel, gt, v0 + horiz([a, b])) for a in grid for b in grid))
    return {"clean": np.mean(clean), "corrupt": np.mean(corrupt), "rec": np.mean(rec),
            "oracle": np.mean(oracle), "nq": np.mean(nqs), "nb": nb}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/s3dis"); ap.add_argument("--n_blocks", type=int, default=16)
    ap.add_argument("--npoints", type=int, default=4096); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--grid_n", type=int, default=13, help="dense tilt grid resolution (queries = grid_n^2)")
    args = ap.parse_args()

    ds = S3DISRooms(args.root, split="test", test_area=5); bb = FrozenSemSeg()
    rows = []
    for s in args.seeds:
        r = run_seed(bb, ds, s, args.n_blocks, args.npoints, grid_n=args.grid_n)
        rows.append(r)
        print(f"  seed {s} (n={r['nb']}): clean {r['clean']:.3f}  corrupt {r['corrupt']:.3f}  "
              f"recover {r['rec']:.3f}  oracle {r['oracle']:.3f}  ({r['nq']:.0f} q/block)", flush=True)

    def ms(key):
        v = np.array([r[key] for r in rows]); return v.mean(), v.std()
    print(f"\n[S3DIS tilt canonicalization, DETERMINISTIC recovery, seeds={args.seeds}, n={args.n_blocks}/seed]")
    for key, lab in [("corrupt", "corrupted (tilt)"), ("rec", "recovered (label-free)"),
                     ("oracle", "oracle (GT)"), ("clean", "clean (upright)")]:
        m, sd = ms(key); print(f"  {lab:24s} = {m:.4f} +/- {sd:.4f}")
    mc, _ = ms("corrupt"); mr, _ = ms("rec"); mo, _ = ms("oracle")
    print(f"  -> recovers {mr-mc:+.4f} mIoU label-free ({100*(mr-mc)/(mo-mc+1e-9):.0f}% of oracle gap), "
          f"std across seeds = {ms('rec')[1]:.4f}")


if __name__ == "__main__":
    main()
