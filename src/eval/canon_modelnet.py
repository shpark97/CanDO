"""Pose recovery on ModelNet40 classification (Table 1, ModelNet40 rows).

yanx27 cls is trained WITHOUT rotation augmentation -> rotation-sensitive (the classic ModelNet
rotation-robustness problem). We corrupt with a random SO(3) rotation and recover, gradient-free and
LABEL-FREE, by maximizing the single max-softmax prob (confidence). Metric = classification accuracy.
This tests whether the canonicalization headroom (shown for part-seg & sem-seg) generalizes to
classification. Recovery is CMA-ES (3-D SO(3), where it wins) SEEDED PER-SHAPE -> deterministic,
seed-robust.

Run:  python -m src.eval.canon_modelnet --n 80
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.modelnet import ModelNet40Cls
from src.downstream.modelnet_cls import FrozenModelNetCls
from src.zoo.canonicalize import Rso3


def build_backbone(name: str):
    """Frozen classifier black-box: 'pointnet2' = PointNet++ (default), 'pointnext' = PointNeXt-S."""
    if name == "pointnext":
        from src.downstream.pointnext_cls import FrozenPointNeXtCls
        return FrozenPointNeXtCls()
    return FrozenModelNetCls()


def pred_at(bb, xyz, v):
    R = Rso3(v)
    return bb((xyz @ R.T).unsqueeze(0)).squeeze(0)  # (40,) log-prob


def confidence(bb, xyz, v):
    return pred_at(bb, xyz, v).exp().max().item()


def recover_cma(bb, xyz, budget=240, sigma0=1.2, restarts=2, seed=0):
    import cma
    best_v, best_c = torch.zeros(3, device=xyz.device), -1.0
    for r in range(restarts):
        x0 = (np.random.default_rng(seed + r).normal(size=3) * 0.3).tolist()
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "maxfevals": budget // restarts, "popsize": 10, "verbose": -9, "seed": seed + r + 1})
        while not es.stop():
            sols = es.ask()
            es.tell(sols, [-confidence(bb, xyz, torch.tensor(x, device=xyz.device, dtype=torch.float32)) for x in sols])
        v = torch.tensor(es.result.xbest, device=xyz.device, dtype=torch.float32)
        c = confidence(bb, xyz, v)
        if c > best_c:
            best_c, best_v = c, v
    return best_v


def main():
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/modelnet_dl"); ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--cseed", type=int, default=0, help="corruption-realization seed offset")
    ap.add_argument("--Moracle", type=int, default=256)
    ap.add_argument("--idx_start", type=int, default=0); ap.add_argument("--idx_end", type=int, default=-1)
    ap.add_argument("--dump", type=str, default="")
    ap.add_argument("--backbone", default="pointnet2",
                    choices=["pointnet2", "pointnext"])
    args = ap.parse_args()

    ds = ModelNet40Cls(args.root, "test", args.npoints); bb = build_backbone(args.backbone)
    rng = np.random.default_rng(args.seed)
    idx = list(range(len(ds))) if args.n < 0 else rng.choice(len(ds), args.n, replace=False).tolist()
    end = len(idx) if args.idx_end < 0 else min(args.idx_end, len(idx))
    shard = idx[args.idx_start:end]
    OFF = args.cseed * 100_000_000                                  # per-index corruption seeding -> shardable
    Z = torch.zeros(3, device="cuda")

    res = {k: [] for k in ["clean", "corrupt", "ours", "oracle"]}
    for j, i in enumerate(shard):
        s = ds[i]; xyz = s["points"].cuda().float(); y = s["label"]
        res["clean"].append(int(pred_at(bb, xyz, Z).argmax().item() == y))
        crng = np.random.default_rng(7_000_000 + i + OFF)
        axis = crng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * crng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        xr = xyz @ Rso3(v0).T
        res["corrupt"].append(int(pred_at(bb, xr, Z).argmax().item() == y))
        v = recover_cma(bb, xr, seed=2_000_000 + i + OFF); res["ours"].append(int(pred_at(bb, xr, v).argmax().item() == y))
        orng = np.random.default_rng(3_000_000 + i + OFF)          # oracle: any candidate correct
        ok = int(pred_at(bb, xr, Z).argmax().item() == y)
        for _ in range(args.Moracle - 1):
            if ok:
                break
            a = orng.normal(size=3); a /= np.linalg.norm(a) + 1e-9
            vv = torch.tensor(a * orng.uniform(0, np.pi), device="cuda", dtype=torch.float32)
            ok = int(pred_at(bb, xr, vv).argmax().item() == y)
        res["oracle"].append(ok)
        if (j + 1) % 200 == 0:
            print(f"  ... {j+1}/{len(shard)}", flush=True)

    if args.dump:
        json.dump({"idx": shard, **res}, open(args.dump, "w"))
    m = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"[ModelNet40 cls SO(3), n={len(shard)} (idx {args.idx_start}:{end}), cseed={args.cseed}]")
    for k in ["clean", "corrupt", "ours", "oracle"]:
        print(f"  {k:8s} acc = {m[k]:.3f}")


if __name__ == "__main__":
    main()
