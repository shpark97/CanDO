"""Competitive baseline: classical PCA canonicalization vs our label-free confidence search.

PCA-canon aligns the cloud's principal axes to the coordinate axes (sign-disambiguated by skewness) —
the standard parameter-free canonicalizer, known to be unreliable (axis/sign ambiguity; PCA axes != the
model's semantic frame). Compared on ShapeNet part-seg under SO(3) corruption. Fully batched (one forward
over M candidate rotations yields both our confidence-selected recovery AND the oracle), so it's fast.

Run:  python -m src.eval.baseline_pca_canon --n 24
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg, make_input
from src.eval.metrics import PART_RANGES, mask_argmax, shape_miou
from src.zoo.canonicalize import Rso3


def miou_one(bb, xyz, nrm, clsv, cls, gt):
    lg = bb(make_input(xyz, nrm, True), clsv).squeeze(0)
    return shape_miou(mask_argmax(lg, cls), gt, cls)


def pca_canon(xyz, nrm):
    c = xyz - xyz.mean(0, keepdim=True)
    cov = c.T @ c / c.shape[0]
    _, V = torch.linalg.eigh(cov); V = V[:, [2, 1, 0]]
    proj = c @ V
    sign = torch.sign((proj ** 3).sum(0)); sign[sign == 0] = 1.0
    V = V * sign
    return c @ V, nrm @ V


def batched(bb, xyz, nrm, cls, vs):
    Rs = torch.stack([Rso3(v) for v in vs])
    xr = torch.einsum("mij,nj->mni", Rs, xyz); nr = torch.einsum("mij,nj->mni", Rs, nrm)
    feat = torch.cat([xr, nr], -1).transpose(1, 2).contiguous()
    lg = bb(feat, torch.full((len(vs),), cls, dtype=torch.long))   # (M,N,50)
    conf = lg.exp()[:, :, PART_RANGES[cls]].max(-1).values.mean(-1)  # (M,)
    return lg, conf


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--M", type=int, default=128); ap.add_argument("--lam", type=float, default=0.08)
    args = ap.parse_args()
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg(); rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), args.n, replace=False).tolist()

    res = {"clean": [], "corrupt": [], "pca": [], "ours": [], "oracle": []}
    for j, i in enumerate(idx):
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        res["clean"].append(miou_one(bb, xyz, nrm, clsv, cls, gt))
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * rng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        R0 = Rso3(v0); xr, nr = xyz @ R0.T, nrm @ R0.T
        res["corrupt"].append(miou_one(bb, xr, nr, clsv, cls, gt))
        xp, npc = pca_canon(xr, nr); res["pca"].append(miou_one(bb, xp, npc, clsv, cls, gt))
        # candidates: identity + random SO(3); one batched forward -> confidence (ours) + per-cand mIoU (oracle)
        crng = np.random.default_rng(100 + j)
        vs = [torch.zeros(3, device="cuda")]; norms = [0.0]
        for _ in range(args.M - 1):
            ax = crng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-9; ang = crng.uniform(0, np.pi)
            vs.append(torch.tensor(ax * ang, device="cuda", dtype=torch.float32)); norms.append(ang)
        lg, conf = batched(bb, xr, nr, cls, vs)
        mious = torch.tensor([shape_miou(mask_argmax(lg[m], cls), gt, cls) for m in range(len(vs))])
        k = int((conf - args.lam * torch.tensor(norms, device="cuda")).argmax().item())
        res["ours"].append(mious[k].item()); res["oracle"].append(mious.max().item())

    m = {k: float(np.mean(v)) for k, v in res.items()}; gap = m["clean"] - m["corrupt"]
    print(f"[PCA-canon vs ours @ ShapeNet part-seg SO(3), n={args.n}]")
    for k in ["clean", "corrupt", "pca", "ours", "oracle"]:
        pct = 100 * (m[k] - m["corrupt"]) / (gap + 1e-9)
        print(f"  {k:8s} = {m[k]:.3f}" + (f"   ({pct:.0f}% gap)" if k in ("pca", "ours", "oracle") else ""))


if __name__ == "__main__":
    main()
