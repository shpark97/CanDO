"""Classical canonicalization baselines on the AZIMUTH setting (ShapeNet part-seg, 1-D).

The comparison table in the paper pits PCA / PCA+conf against our search under SO(3) only. Here we show
the same baselines also fail on the azimuth corruption that our method recovers 76% of the oracle gap on,
so the baseline weakness is not specific to full SO(3). PCA-canon aligns principal axes (sign-disambiguated
by skewness); PCA+conf additionally picks, among the four cardinal yaw flips of the PCA frame, the one with
highest model confidence. Ours is the label-free coarse-scan + SPSA azimuth search.

Run:  PYTHONPATH=. python -m src.eval.baseline_azimuth --n 48
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg, make_input
from src.eval.metrics import mask_argmax, shape_miou
from src.eval.baseline_pca_canon import miou_one, pca_canon
from src.zoo.canonicalize import Rz, confidence, zoo_canonicalize


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--cseed", type=int, default=0, help="corruption-realization seed offset")
    ap.add_argument("--idx_start", type=int, default=0); ap.add_argument("--idx_end", type=int, default=-1)
    ap.add_argument("--dump", type=str, default="")
    args = ap.parse_args()
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg(); rng = np.random.default_rng(args.seed)
    idx = list(range(len(ds))) if args.n < 0 else rng.choice(len(ds), args.n, replace=False).tolist()
    end = len(idx) if args.idx_end < 0 else min(args.idx_end, len(idx))
    shard = idx[args.idx_start:end]
    OFF = args.cseed * 100_000_000

    res = {"clean": [], "corrupt": [], "pca": [], "pca_conf": [], "ours": [], "oracle": []}
    grid = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    for j, i in enumerate(shard):
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        res["clean"].append(miou_one(bb, xyz, nrm, clsv, cls, gt))
        a0 = np.random.default_rng(7_000_000 + i + OFF).uniform(0, 2 * np.pi)   # per-index corruption
        R0 = Rz(torch.tensor([a0], device="cuda", dtype=torch.float32))
        xr, nr = xyz @ R0.T, nrm @ R0.T
        res["corrupt"].append(miou_one(bb, xr, nr, clsv, cls, gt))

        def miou_at(angle, base_x=xr, base_n=nr):
            R = Rz(torch.tensor([angle], device="cuda", dtype=torch.float32))
            lg = bb(make_input(base_x @ R.T, base_n @ R.T, True), clsv).squeeze(0)
            return shape_miou(mask_argmax(lg, cls), gt, cls)

        # PCA-canon
        xp, npc = pca_canon(xr, nr); res["pca"].append(miou_one(bb, xp, npc, clsv, cls, gt))
        # PCA + confidence: best of 4 cardinal yaw flips of the PCA frame, chosen by confidence
        best_c, best_m = -1.0, 0.0
        for q in range(4):
            ang = q * np.pi / 2
            c = confidence(bb, xp, npc, clsv, torch.tensor([ang], device="cuda", dtype=torch.float32)).item()
            if c > best_c:
                best_c = c
                R = Rz(torch.tensor([ang], device="cuda", dtype=torch.float32))
                best_m = miou_one(bb, xp @ R.T, npc @ R.T, clsv, cls, gt)
        res["pca_conf"].append(best_m)
        # ours: label-free azimuth search
        ang, _ = zoo_canonicalize(bb, xr, nr, clsv, coarse=16, steps=args.steps, seed=2_000_000 + i + OFF)
        res["ours"].append(miou_at(ang.item()))
        res["oracle"].append(max(miou_at(a) for a in grid))
        if (j + 1) % 100 == 0:
            print(f"  ... {j+1}/{len(shard)}", flush=True)

    if args.dump:
        json.dump({"idx": shard, **res}, open(args.dump, "w"))
    m = {k: float(np.mean(v)) for k, v in res.items()}; gap = m["clean"] - m["corrupt"] + 1e-9
    print(f"[Azimuth baselines @ ShapeNet part-seg, n={len(shard)} (idx {args.idx_start}:{end}), cseed={args.cseed}]")
    for k in ["clean", "corrupt", "pca", "pca_conf", "ours", "oracle"]:
        pct = 100 * (m[k] - m["corrupt"]) / gap
        tag = f"   ({pct:.0f}% gap)" if k in ("pca", "pca_conf", "ours", "oracle") else ""
        print(f"  {k:9s} = {m[k]:.3f}{tag}")


if __name__ == "__main__":
    main()
