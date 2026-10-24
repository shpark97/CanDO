"""Pose recovery on the FROZEN PointNeXt-S ShapeNetPart part-seg model (architecture-independence of
part-seg recovery). Mirrors the PointNet++ part-seg protocol (baseline_azimuth / baseline_compare_partseg):
label-free max-confidence recovery with the identity kept as a candidate, category-masked instance mIoU,
256-candidate same-family oracle, per-index corruption seeding so any shard reproduces the single-process
numbers. Supports azimuth (1-D coarse-scan+SPSA) and full SO(3) (3-D CMA-ES).

Run (single GPU, full test, one corruption seed):
  PYTHONPATH=third_party/PointNeXt:. python -m src.eval.canon_partseg_pnx --transform so3 --n -1 --cseed 0
8-GPU shard: pass --idx_start/--idx_end/--dump per GPU, then aggregate the dumps.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnext_partseg import FrozenPointNeXtPartSeg
from src.downstream.pointnet2_partseg import make_input
from src.eval.metrics import mask_argmax, shape_miou
from src.zoo.canonicalize import (
    Rz, Rso3, confidence, confidence_so3, zoo_canonicalize, zoo_canonicalize_so3_cma)


def miou_az(bb, xr, nr, clsv, cls, gt, angle):
    R = Rz(torch.tensor([angle], device=xr.device, dtype=torch.float32))
    lg = bb(make_input(xr @ R.T, nr @ R.T, True), clsv).squeeze(0)
    return shape_miou(mask_argmax(lg, cls), gt, cls)


def miou_so3(bb, xr, nr, clsv, cls, gt, v):
    R = Rso3(v)
    lg = bb(make_input(xr @ R.T, nr @ R.T, True), clsv).squeeze(0)
    return shape_miou(mask_argmax(lg, cls), gt, cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transform", choices=["azimuth", "so3"], required=True)
    ap.add_argument("--n", type=int, default=-1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cseed", type=int, default=0, help="corruption-realization seed offset")
    ap.add_argument("--idx_start", type=int, default=0); ap.add_argument("--idx_end", type=int, default=-1)
    ap.add_argument("--dump", type=str, default=""); ap.add_argument("--Moracle", type=int, default=256)
    ap.add_argument("--steps", type=int, default=15); ap.add_argument("--budget", type=int, default=300)
    ap.add_argument("--ckpt", type=str, default="")
    args = ap.parse_args()

    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test",
                      npoints=None, normalize=False)  # RAW points; adapter reproduces PointNeXt transform
    bb = FrozenPointNeXtPartSeg(**({"ckpt": args.ckpt} if args.ckpt else {}))
    rng = np.random.default_rng(args.seed)
    idx = list(range(len(ds))) if args.n < 0 else rng.choice(len(ds), args.n, replace=False).tolist()
    end = len(idx) if args.idx_end < 0 else min(args.idx_end, len(idx))
    shard = idx[args.idx_start:end]
    OFF = args.cseed * 100_000_000

    res = {"clean": [], "corrupt": [], "ours": [], "oracle": []}
    for j, i in enumerate(shard):
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        crng = np.random.default_rng(7_000_000 + i + OFF)
        orng = np.random.default_rng(3_000_000 + i + OFF)
        rseed = 2_000_000 + i + OFF

        if args.transform == "azimuth":
            res["clean"].append(miou_az(bb, xyz, nrm, clsv, cls, gt, 0.0))
            a0 = crng.uniform(0, 2 * np.pi)
            R0 = Rz(torch.tensor([a0], device="cuda", dtype=torch.float32))
            xr, nr = xyz @ R0.T, nrm @ R0.T
            res["corrupt"].append(miou_az(bb, xr, nr, clsv, cls, gt, 0.0))
            ang, _ = zoo_canonicalize(bb, xr, nr, clsv, coarse=16, steps=args.steps, seed=rseed)
            # identity kept as a candidate (lambda=0): pick higher confidence
            c_id = confidence(bb, xr, nr, clsv, torch.zeros(1, device="cuda")).item()
            c_an = confidence(bb, xr, nr, clsv, ang.view(1)).item()
            sel = ang.item() if c_an > c_id else 0.0
            res["ours"].append(miou_az(bb, xr, nr, clsv, cls, gt, sel))
            cand = [0.0] + [orng.uniform(0, 2 * np.pi) for _ in range(args.Moracle - 1)]
            res["oracle"].append(max(miou_az(bb, xr, nr, clsv, cls, gt, a) for a in cand))
        else:
            res["clean"].append(miou_so3(bb, xyz, nrm, clsv, cls, gt, torch.zeros(3, device="cuda")))
            axis = crng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
            v0 = torch.tensor(axis * crng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
            R0 = Rso3(v0); xr, nr = xyz @ R0.T, nrm @ R0.T
            res["corrupt"].append(miou_so3(bb, xr, nr, clsv, cls, gt, torch.zeros(3, device="cuda")))
            v, _ = zoo_canonicalize_so3_cma(bb, xr, nr, clsv, budget=args.budget, restarts=2, seed=rseed)
            c_id = confidence_so3(bb, xr, nr, clsv, torch.zeros(3, device="cuda")).item()
            c_v = confidence_so3(bb, xr, nr, clsv, v).item()
            sel = v if c_v > c_id else torch.zeros(3, device="cuda")
            res["ours"].append(miou_so3(bb, xr, nr, clsv, cls, gt, sel))
            best = miou_so3(bb, xr, nr, clsv, cls, gt, torch.zeros(3, device="cuda"))
            for _ in range(args.Moracle - 1):
                a = orng.normal(size=3); a /= np.linalg.norm(a) + 1e-9
                vv = torch.tensor(a * orng.uniform(0, np.pi), device="cuda", dtype=torch.float32)
                best = max(best, miou_so3(bb, xr, nr, clsv, cls, gt, vv))
            res["oracle"].append(best)
        if (j + 1) % 50 == 0:
            print(f"  ... {j+1}/{len(shard)}", flush=True)

    if args.dump:
        json.dump({"idx": shard, **res}, open(args.dump, "w"))
    m = {k: float(np.mean(v)) for k, v in res.items()}; gap = m["oracle"] - m["corrupt"] + 1e-9
    print(f"[PointNeXt-S part-seg {args.transform} recovery, n={len(shard)} "
          f"(idx {args.idx_start}:{end}), cseed={args.cseed}]")
    for k in ["clean", "corrupt", "ours", "oracle"]:
        tag = f"   ({100*(m[k]-m['corrupt'])/gap:.0f}% gap)" if k in ("ours", "oracle") else ""
        print(f"  {k:8s} = {m[k]:.3f}{tag}")


if __name__ == "__main__":
    main()
