"""Gradient-free, label-free tilt canonicalization on S3DIS scenes.

The frozen S3DIS sem-seg model is z-rotation augmented (yaw-robust) but assumes gravity / z-up:
a pitch/roll tilt (uneven floor, miscalibrated sensor) collapses it (mIoU 0.63 -> 0.09 at 60deg).
This is the scene analog of the ShapeNet rotation room. We recover the upright frame by
gradient-free optimization of a 2-D tilt (axis-angle v=(a,b,0)) against a BLACK-BOX, LABEL-FREE
objective (mean max-softmax / confidence). No GT, no model gradients.

Any derivative-free optimizer applies. The landscape is 2-D
and multi-modal-ish, so we compare a global population ES (CMA-ES) vs local SPSA vs random search
-- the same estimator comparison as on ShapeNet SO(3), now at scene scale.

Run:  python -m src.eval.canon_s3dis_tilt --n_blocks 24
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.s3dis import S3DISRooms, sample_block
from src.downstream.pointnet2_semseg import FrozenSemSeg
from src.eval.canon_s3dis import block_miou
from src.samplers.classic import fps_sample
from src.zoo.canonicalize import Rso3
from src.zoo.estimator import SPSA


def tilt_input(block, sel, v: torch.Tensor) -> torch.Tensor:
    """(1,9,K) sem-seg input with the block rotated by axis-angle v about its center (z=block mean)."""
    coord, color = block["coord"][sel], block["color"][sel]
    c = block["center"].clone(); c[2] = coord[:, 2].mean()
    rmax = block["room_max"]
    d = (coord - c) @ Rso3(v).T
    ac = d + c
    f = torch.empty((coord.shape[0], 9), device=coord.device)
    f[:, 0] = d[:, 0]; f[:, 1] = d[:, 1]; f[:, 2] = ac[:, 2]
    f[:, 3:6] = color / 255.0
    f[:, 6] = ac[:, 0] / (rmax[0] + 1e-8)
    f[:, 7] = ac[:, 1] / (rmax[1] + 1e-8)
    f[:, 8] = ac[:, 2] / (rmax[2] + 1e-8)
    return f.t().unsqueeze(0).contiguous()


def confidence(bb, block, sel, v) -> float:
    return bb(tilt_input(block, sel, v)).squeeze(0).exp().max(-1).values.mean().item()


def miou(bb, block, sel, gt, v) -> float:
    return block_miou(bb(tilt_input(block, sel, v)).squeeze(0).argmax(-1), gt)


def horiz(v2):
    """2-vector (a,b) -> axis-angle 3-vector (a,b,0): tilt about a horizontal axis (yaw is free)."""
    return torch.tensor([float(v2[0]), float(v2[1]), 0.0], device="cuda", dtype=torch.float32)


def recover_cma(bb, block, sel, v0, budget=160, sigma0=0.5, restarts=2, seed=0):
    import cma
    best_w, best_c, nq = torch.zeros(2), -1.0, 0
    for r in range(restarts):
        x0 = (np.random.default_rng(seed + r).normal(size=2) * 0.2).tolist()
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "maxfevals": budget // restarts, "popsize": 8, "verbose": -9, "seed": seed + r + 1})
        while not es.stop():
            sols = es.ask(); fits = []
            for x in sols:
                fits.append(-confidence(bb, block, sel, v0 + horiz(x))); nq += 1
            es.tell(sols, fits)
        c = confidence(bb, block, sel, v0 + horiz(es.result.xbest)); nq += 1
        if c > best_c:
            best_c, best_w = c, torch.tensor(es.result.xbest)
    return v0 + horiz(best_w), nq


def recover_spsa(bb, block, sel, v0, restarts=4, steps=18, lr=0.25, c=0.1, seed=0):
    rng = np.random.default_rng(seed); est = SPSA(c=c, avg=1)
    neg = lambda w: -torch.tensor(confidence(bb, block, sel, v0 + horiz(w)))  # noqa: E731
    best_w, best_c, nq = torch.zeros(2), -1.0, 0
    for _ in range(restarts):
        w = torch.tensor(rng.normal(size=2) * 0.3, dtype=torch.float32)
        for t in range(steps):
            g = est.estimate(w, neg); w = w - (lr / (1 + 0.1 * t)) * g; nq += est.last_n_evals
        cc = confidence(bb, block, sel, v0 + horiz(w)); nq += 1
        if cc > best_c:
            best_c, best_w = cc, w.clone()
    return v0 + horiz(best_w), nq


def recover_random(bb, block, sel, v0, n_queries, seed=0):
    rng = np.random.default_rng(seed); best_w, best_c = torch.zeros(2), -1.0
    for _ in range(n_queries):
        w = torch.tensor(rng.uniform(-1.0, 1.0, size=2), dtype=torch.float32)
        c = confidence(bb, block, sel, v0 + horiz(w))
        if c > best_c:
            best_c, best_w = c, w
    return v0 + horiz(best_w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/s3dis"); ap.add_argument("--n_blocks", type=int, default=24)
    ap.add_argument("--npoints", type=int, default=4096); ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ds = S3DISRooms(args.root, split="test", test_area=5); bb = FrozenSemSeg()
    g = torch.Generator(device="cuda"); g.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    grid = np.linspace(-0.9, 0.9, 7)
    Z = torch.zeros(3, device="cuda")

    res = {"clean": [], "corrupt": [], "spsa": [], "cma": [], "rand": [], "oracle": []}
    q = {"spsa": [], "cma": []}
    corr_pairs = []  # (confidence, mIoU) over the grid, for the enabling-condition check
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
        res["clean"].append(miou(bb, block, sel, gt, Z))
        # random tilt corruption: horizontal axis, angle 0.3-0.9 rad (~17-52 deg)
        ang = rng.uniform(0.3, 0.9); phi = rng.uniform(0, 2 * np.pi)
        v0 = horiz([np.cos(phi) * ang, np.sin(phi) * ang])
        res["corrupt"].append(miou(bb, block, sel, gt, v0))

        vS, qS = recover_spsa(bb, block, sel, v0, seed=nb); res["spsa"].append(miou(bb, block, sel, gt, vS)); q["spsa"].append(qS)
        vC, qC = recover_cma(bb, block, sel, v0, seed=nb); res["cma"].append(miou(bb, block, sel, gt, vC)); q["cma"].append(qC)
        budget = int(np.mean([qS, qC]))
        vR = recover_random(bb, block, sel, v0, budget, seed=nb); res["rand"].append(miou(bb, block, sel, gt, vR))
        # oracle (best residual tilt by GT) + correlation pairs
        best = -1.0
        for a in grid:
            for b in grid:
                vv = v0 + horiz([a, b])
                cf = confidence(bb, block, sel, vv); mm = miou(bb, block, sel, gt, vv)
                corr_pairs.append((cf, mm)); best = max(best, mm)
        res["oracle"].append(best)

    m = {k: float(np.mean(v)) for k, v in res.items()}
    cp = np.array(corr_pairs); corr = float(np.corrcoef(cp[:, 0], cp[:, 1])[0, 1])
    gap = m["clean"] - m["corrupt"]
    print(f"[tilt canonicalization @ S3DIS Area_5, n={nb} blocks, {args.npoints} pts]")
    print(f"  clean (upright)        mIoU = {m['clean']:.4f}")
    print(f"  corrupted (random tilt)mIoU = {m['corrupt']:.4f}")
    print(f"  SPSA       (label-free)     = {m['spsa']:.4f}   ({np.mean(q['spsa']):.0f} q/block)")
    print(f"  CMA-ES     (label-free)     = {m['cma']:.4f}   ({np.mean(q['cma']):.0f} q/block)")
    print(f"  random-search (budget-match)= {m['rand']:.4f}")
    print(f"  oracle (7x7 grid, GT)  mIoU = {m['oracle']:.4f}")
    best = max(m['spsa'], m['cma'])
    print(f"  -> best optimizer recovers {best-m['corrupt']:+.4f} mIoU label-free "
          f"({100*(best-m['corrupt'])/(gap+1e-9):.0f}% of clean gap)")
    print(f"  enabling condition: corr(confidence, mIoU) over tilt grid = {corr:+.3f}")


if __name__ == "__main__":
    main()
