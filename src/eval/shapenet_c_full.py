"""Full ShapeNet-C (part-seg): all 7 official atomic corruptions x 5 severities on the full test set.

Label- and normal-aware so that density (dropout) and addition corruptions keep per-point labels aligned
(added points are excluded from mIoU; dropped points keep their labels). Compares source (no recovery)
vs ours (label-free batched-random SO(3) canonicalization + identity prior). Writes per-corruption
results incrementally so a long run is robust to interruption.

Run:  python -m src.eval.shapenet_c_full --n -1            (full test)
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg
from src.eval.metrics import PART_RANGES, mask_argmax, shape_miou
from src.zoo.canonicalize import Rso3

CORRUPTIONS = ["rotate", "scale", "jitter", "dropout_global", "dropout_local", "add_global", "add_local"]
FAMILY = {"rotate": "transform", "scale": "transform", "jitter": "noise", "add_global": "noise",
          "add_local": "noise", "dropout_global": "density", "dropout_local": "density"}


def _rotmat(a):
    Rx = np.array([[1, 0, 0], [0, np.cos(a[0]), -np.sin(a[0])], [0, np.sin(a[0]), np.cos(a[0])]])
    Ry = np.array([[np.cos(a[1]), 0, np.sin(a[1])], [0, 1, 0], [-np.sin(a[1]), 0, np.cos(a[1])]])
    Rz = np.array([[np.cos(a[2]), -np.sin(a[2]), 0], [np.sin(a[2]), np.cos(a[2]), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def corrupt(xyz, nrm, seg, ctype, level, rng):
    """Returns (xyz2, nrm2, seg2, valid2) on the device of xyz. valid2 marks GT-labeled points."""
    N = xyz.shape[0]; dev = xyz.device
    valid = torch.ones(N, dtype=torch.bool, device=dev)
    if ctype == "rotate":
        ac = math.pi / 6 / 5 * (level + 1); a = rng.uniform(-ac, ac, size=3)
        R = torch.tensor(_rotmat(a), device=dev, dtype=torch.float32)
        return xyz @ R.T, nrm @ R.T, seg, valid
    if ctype == "scale":
        s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
        f = torch.tensor(rng.uniform(1. / s, s, size=3), device=dev, dtype=torch.float32)
        xs = xyz * f; xs = xs / (xs.norm(dim=1).max() + 1e-9)
        return xs, torch.nn.functional.normalize(nrm / f, dim=1), seg, valid
    if ctype == "jitter":
        sigma = 0.01 * (level + 1)
        return xyz + sigma * torch.randn_like(xyz), nrm, seg, valid
    if ctype == "dropout_global":
        drop = [0.25, 0.375, 0.5, 0.625, 0.75][level]
        keep = int(N * (1 - drop)); idx = torch.tensor(rng.permutation(N)[:keep], device=dev)
        return xyz[idx], nrm[idx], seg[idx], valid[idx]
    if ctype == "dropout_local":
        total = 100 * (level + 1); nc = int(rng.integers(1, 8)); cur = N
        keep_mask = torch.ones(N, dtype=torch.bool, device=dev)
        sizes = (np.ones(nc) * total / nc).astype(int)
        for K in sizes:
            avail = keep_mask.nonzero(as_tuple=False).squeeze(1)
            if avail.numel() <= K:
                break
            c0 = xyz[avail[int(rng.integers(avail.numel()))]]
            d = ((xyz[avail] - c0) ** 2).sum(1)
            far = avail[d.argsort()[-int(K):]]
            keep_mask[far] = False
        return xyz[keep_mask], nrm[keep_mask], seg[keep_mask], valid[keep_mask]
    if ctype in ("add_global", "add_local"):
        npts = 10 * (level + 1) if ctype == "add_global" else 100 * (level + 1)
        if ctype == "add_global":
            add = torch.tensor(_sample_sphere(npts, rng), device=dev, dtype=torch.float32)
        else:
            nc = int(rng.integers(1, 8)); chunks = []
            for _ in range(nc):
                c0 = xyz[int(rng.integers(N))].cpu().numpy()
                chunks.append(c0 + rng.uniform(0.075, 0.125) * rng.standard_normal((npts // nc + 1, 3)))
            add = torch.tensor(np.concatenate(chunks)[:npts], device=dev, dtype=torch.float32)
        xyz2 = torch.cat([xyz, add]); nrm2 = torch.cat([nrm, torch.nn.functional.normalize(torch.randn_like(add), dim=1)])
        seg2 = torch.cat([seg, torch.zeros(add.shape[0], dtype=seg.dtype, device=dev)])
        valid2 = torch.cat([valid, torch.zeros(add.shape[0], dtype=torch.bool, device=dev)])
        return xyz2, nrm2, seg2, valid2
    raise ValueError(ctype)


def _sample_sphere(n, rng):
    p = rng.standard_normal((n * 3, 3)); p = p[np.einsum("ij,ij->i", p, p) <= 1][:n]
    while p.shape[0] < n:
        q = rng.standard_normal((n, 3)); q = q[np.einsum("ij,ij->i", q, q) <= 1]; p = np.concatenate([p, q])
    return p[:n]


def miou_valid(bb, xyz, nrm, cls, seg, valid):
    lg = bb(torch.cat([xyz, nrm], -1).t().unsqueeze(0).contiguous(), torch.tensor([cls])).squeeze(0)
    pred = mask_argmax(lg, cls)
    return shape_miou(pred[valid], seg[valid], cls)


def recover(bb, xyz, nrm, cls, seg, valid, M=64, lam=0.4, seed=0):
    rng = np.random.default_rng(seed)
    vs = [torch.zeros(3, device=xyz.device)]; norms = [0.0]
    for _ in range(M - 1):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-9; ang = rng.uniform(0, np.pi)
        vs.append(torch.tensor(ax * ang, device=xyz.device, dtype=torch.float32)); norms.append(ang)
    Rs = torch.stack([Rso3(v) for v in vs])
    xr = torch.einsum("mij,nj->mni", Rs, xyz); nr = torch.einsum("mij,nj->mni", Rs, nrm)
    feat = torch.cat([xr, nr], -1).transpose(1, 2).contiguous()
    lg = bb(feat, torch.full((len(vs),), cls, dtype=torch.long))     # (M,N,50)
    conf = lg.exp()[:, :, PART_RANGES[cls]].max(-1).values.mean(-1)
    k = int((conf - lam * torch.tensor(norms, device=xyz.device)).argmax().item())
    return shape_miou(mask_argmax(lg[k], cls)[valid], seg[valid], cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    ap.add_argument("--n", type=int, default=-1); ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--lam", type=float, default=0.4); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/shapenet_c_full.out")
    args = ap.parse_args()
    ds = ShapeNetPart(args.root, "test", npoints=2048); bb = FrozenPartSeg()
    rng0 = np.random.default_rng(args.seed)
    idx = list(range(len(ds))) if args.n < 0 else rng0.choice(len(ds), args.n, replace=False).tolist()
    shapes = [ds[i] for i in idx]
    with open(args.out, "a") as f:
        f.write(f"[Full ShapeNet-C part-seg, n={len(idx)}, M={args.M}, lam={args.lam}]\n"); f.flush()
        rows = []
        for ctype in CORRUPTIONS:
            src, our = [], []
            for lvl in range(5):
                for sj, s in enumerate(shapes):
                    xyz, nrm, seg, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
                    crng = np.random.default_rng(1000 * lvl + sj)
                    xc, nc, sc, vc = corrupt(xyz, nrm, seg, ctype, lvl, crng)
                    src.append(miou_valid(bb, xc, nc, cls, sc, vc))
                    our.append(recover(bb, xc, nc, cls, sc, vc, M=args.M, lam=args.lam, seed=7000 + sj))
            so, oo = float(np.mean(src)), float(np.mean(our))
            rows.append((ctype, FAMILY[ctype], so, oo))
            f.write(f"  {ctype:16s} ({FAMILY[ctype]:9s}): source {so:.3f}  ours {oo:.3f}  d {oo-so:+.3f}\n"); f.flush()
        for fam in ["transform", "noise", "density"]:
            r = [x for x in rows if x[1] == fam]
            f.write(f"  {fam:9s} mean: source {np.mean([x[2] for x in r]):.3f}  ours {np.mean([x[3] for x in r]):.3f}\n")
        f.write(f"  ALL7 mean: source {np.mean([x[2] for x in rows]):.3f}  ours {np.mean([x[3] for x in rows]):.3f}\n")
        f.write("SHAPENETC_FULL_DONE\n"); f.flush()


if __name__ == "__main__":
    main()
