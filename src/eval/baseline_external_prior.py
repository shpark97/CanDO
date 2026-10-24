"""External-model canonicalization baseline (ShapeNet part-seg, SO(3)).

A controlled stand-in for the external-prior line (FoCal/CloudFixer): instead of the task model's own
confidence, the canonicalizing objective is the confidence of a SEPARATE frozen model, a ModelNet40
classifier, scoring each candidate pose by how "typical" it looks to that external model. We then evaluate
the frozen part-seg model at the selected pose. This occupies the ext.model=YES row of the comparison with
real numbers, and tests whether an external model's prior beats the self-contained objective. Same per-index
SO(3) corruption as baseline_compare_partseg, so the shapes match. Shardable via --idx_start/--idx_end.

Run:  PYTHONPATH=. python -m src.eval.baseline_external_prior --n -1 --idx_start 0 --idx_end -1 --dump out.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.modelnet_cls import FrozenModelNetCls
from src.downstream.pointnet2_partseg import FrozenPartSeg
from src.eval.baseline_compare_partseg import batched_conf, miou_one, ours_cma
from src.eval.metrics import mask_argmax, shape_miou
from src.samplers.classic import fps_sample
from src.zoo.canonicalize import Rso3


def normalize_unit(p):
    p = p - p.mean(0, keepdim=True)
    return p / (p.norm(dim=1).max() + 1e-9)


def external_canon(ext, fps_pts, vs, norms, lam, chunk=64):
    """Pick the rotation maximizing the EXTERNAL classifier's confidence minus the identity prior."""
    best_score, best_k = -1e9, 0
    for c in range(0, len(vs), chunk):
        Rs = torch.stack([Rso3(v) for v in vs[c:c + chunk]])
        batch = torch.einsum("mij,nj->mni", Rs, fps_pts)            # (m,1024,3)
        conf = ext(batch).softmax(-1).max(-1).values                # (m,)
        score = conf - lam * torch.tensor(norms[c:c + chunk], device=fps_pts.device)
        k = int(score.argmax().item())
        if score[k].item() > best_score:
            best_score = score[k].item(); best_k = c + k
    return vs[best_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=-1); ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--lam", type=float, default=0.08); ap.add_argument("--Moracle", type=int, default=256)
    ap.add_argument("--idx_start", type=int, default=0); ap.add_argument("--idx_end", type=int, default=-1)
    ap.add_argument("--dump", type=str, default="")
    args = ap.parse_args()
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg(); ext = FrozenModelNetCls()
    idx = list(range(len(ds)))
    end = len(idx) if args.idx_end < 0 else min(args.idx_end, len(idx))
    shard = idx[args.idx_start:end]

    res = {k: [] for k in ["clean", "corrupt", "extprior", "ours", "oracle"]}
    for i in shard:
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        res["clean"].append(miou_one(bb, xyz, nrm, clsv, cls, gt))
        crng0 = np.random.default_rng(7_000_000 + i)                # same corruption as the main comparison
        axis = crng0.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * crng0.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        R0 = Rso3(v0); xr, nr = xyz @ R0.T, nrm @ R0.T
        res["corrupt"].append(miou_one(bb, xr, nr, clsv, cls, gt))
        # external-prior canonicalization: candidates incl. identity, scored by the cls model
        crng = np.random.default_rng(5_000_000 + i)
        vs = [torch.zeros(3, device="cuda")]; norms = [0.0]
        for _ in range(args.M - 1):
            a = crng.normal(size=3); a /= np.linalg.norm(a) + 1e-9; ang = crng.uniform(0, np.pi)
            vs.append(torch.tensor(a * ang, device="cuda", dtype=torch.float32)); norms.append(ang)
        fps_pts = normalize_unit(xr[fps_sample(xr, 1024)])
        ve = external_canon(ext, fps_pts, vs, norms, args.lam); Re = Rso3(ve)
        res["extprior"].append(miou_one(bb, xr @ Re.T, nr @ Re.T, clsv, cls, gt))
        vo = ours_cma(bb, xr, nr, cls, seed=2_000_000 + i); Ro = Rso3(vo)
        res["ours"].append(miou_one(bb, xr @ Ro.T, nr @ Ro.T, clsv, cls, gt))
        orng = np.random.default_rng(3_000_000 + i)
        ovs = [torch.zeros(3, device="cuda")]
        for _ in range(args.Moracle - 1):
            a = orng.normal(size=3); a /= np.linalg.norm(a) + 1e-9
            ovs.append(torch.tensor(a * orng.uniform(0, np.pi), device="cuda", dtype=torch.float32))
        omax = 0.0
        for c in range(0, len(ovs), 64):
            lg, _ = batched_conf(bb, xr, nr, cls, ovs[c:c + 64])
            omax = max(omax, max(shape_miou(mask_argmax(lg[m], cls), gt, cls) for m in range(lg.shape[0])))
        res["oracle"].append(omax)

    if args.dump:
        json.dump({"idx": shard, **res}, open(args.dump, "w"))
    m = {k: float(np.mean(v)) for k, v in res.items()}; gap = m["clean"] - m["corrupt"] + 1e-9
    print(f"[external-prior canon @ ShapeNet part-seg SO(3), n={len(shard)} (idx {args.idx_start}:{end})]")
    for k in ["corrupt", "extprior", "ours", "oracle"]:
        print(f"  {k:9s} = {m[k]:.3f}   ({100*(m[k]-m['corrupt'])/gap:+.0f}% gap)")


if __name__ == "__main__":
    main()
