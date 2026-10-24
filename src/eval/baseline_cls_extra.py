"""Table 2, ModelNet40 column: PCA canonicalization, PCA+confidence, and the oracle on ModelNet40
classification under SO(3) corruption (source, TENT, MEMO, and ours come from baseline_memo). Shardable across GPUs via --idx_start/--idx_end.

Run:  PYTHONPATH=. python -m src.eval.baseline_cls_extra --n -1 --idx_start 0 --idx_end -1 --dump out.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.data.modelnet import ModelNet40Cls
from src.downstream.modelnet_cls import FrozenModelNetCls
from src.eval.canon_modelnet import pred_at
from src.zoo.canonicalize import Rso3


def pca_frame(xyz):
    """Principal-axis frame, sign-disambiguated by skewness (as in the part-seg PCA baseline)."""
    c = xyz - xyz.mean(0, keepdim=True)
    cov = c.T @ c / c.shape[0]
    _, V = torch.linalg.eigh(cov); V = V[:, [2, 1, 0]]
    proj = c @ V
    sign = torch.sign((proj ** 3).sum(0)); sign[sign == 0] = 1.0
    V = V * sign
    if torch.det(V) < 0:                       # keep a proper rotation
        V[:, 2] = -V[:, 2]
    return c @ V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/modelnet_dl"); ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--npoints", type=int, default=1024); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=256, help="oracle candidate pool")
    ap.add_argument("--idx_start", type=int, default=0); ap.add_argument("--idx_end", type=int, default=-1)
    ap.add_argument("--dump", type=str, default="")
    args = ap.parse_args()
    ds = ModelNet40Cls(args.root, "test", args.npoints); bb = FrozenModelNetCls()
    idx = list(range(len(ds)))
    end = len(idx) if args.idx_end < 0 else min(args.idx_end, len(idx))
    shard = idx[args.idx_start:end]
    Z = torch.zeros(3, device="cuda")

    res = {k: [] for k in ["source", "pca", "pcaconf", "oracle"]}
    for i in shard:
        s = ds[i]; xyz = s["points"].cuda().float(); y = s["label"]
        crng = np.random.default_rng(7_000_000 + i)                 # per-index SO(3) corruption
        axis = crng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * crng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        xc = xyz @ Rso3(v0).T
        res["source"].append(int(pred_at(bb, xc, Z).argmax().item() == y))
        xp = pca_frame(xc)
        res["pca"].append(int(bb(xp.unsqueeze(0)).squeeze(0).argmax().item() == y))
        # PCA+conf: best of the 4 proper sign-flips of the PCA frame, chosen by model confidence
        best_c, best_ok = -1.0, 0
        flips = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]   # det=+1
        for fx, fy, fz in flips:
            xpf = xp * torch.tensor([fx, fy, fz], device="cuda", dtype=torch.float32)
            lg = bb(xpf.unsqueeze(0)).squeeze(0)
            c = lg.exp().max().item()
            if c > best_c:
                best_c = c; best_ok = int(lg.argmax().item() == y)
        res["pcaconf"].append(best_ok)
        # oracle: is any candidate rotation classified correctly? (recoverable ceiling)
        orng = np.random.default_rng(3_000_000 + i)
        ok = int(pred_at(bb, xc, Z).argmax().item() == y)
        for _ in range(args.M - 1):
            a = orng.normal(size=3); a /= np.linalg.norm(a) + 1e-9
            v = torch.tensor(a * orng.uniform(0, np.pi), device="cuda", dtype=torch.float32)
            if pred_at(bb, xc, v).argmax().item() == y:
                ok = 1; break
        res["oracle"].append(ok)

    if args.dump:
        json.dump({"idx": shard, **res}, open(args.dump, "w"))
    m = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"[cls-extra @ ModelNet40 SO(3), n={len(shard)} (idx {args.idx_start}:{end})]")
    for k in ["source", "pca", "pcaconf", "oracle"]:
        print(f"  {k:8s} acc = {m[k]:.3f}")


if __name__ == "__main__":
    main()
