"""Probe a controlled-augmentation PART SEGMENTATION checkpoint under matched corruptions.

Table 4 evaluation for the train_partseg_aug.py checkpoints.
For a given trained model and a corruption family in {yaw, tilt, so3} we report, on a fixed
ShapeNetPart test subset: clean mIoU, corrupted mIoU (a per-shape random rotation of that family),
recovered mIoU (label-free matched-family candidate search scored by masked confidence with the
identity prior), and the pool oracle (best true mIoU among the candidates).

Run:  PYTHONPATH=. python -m src.eval.eval_partseg_aug --ckpt <path> --corrupt yaw --dump out.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg
from src.eval.metrics import PART_RANGES, mask_argmax, shape_miou
from src.eval.train_aug_ablation import rand_rot

LAM = 0.05


def cand_pool(M, regime, device, seed):
    """Identity + (M-1) matched-family candidate rotations; returns (Rs[M,3,3], angles[M])."""
    rng = np.random.default_rng(seed)
    Rs = [torch.eye(3, device=device)]; ang = [0.0]
    for _ in range(M - 1):
        R = rand_rot(1, regime, device, rng)[0]
        Rs.append(R)
        ang.append(float(np.arccos(np.clip((torch.diagonal(R).sum().item() - 1) / 2, -1, 1))))
    return torch.stack(Rs), torch.tensor(ang, device=device)


def pool_eval(bb, xyz, nrm, cls, gt, Rs, ang, chunk=32):
    """mIoU of every candidate rotation plus its masked confidence; returns (mious[M], conf[M])."""
    clsv = torch.full((1,), cls, dtype=torch.long)
    mious, confs = [], []
    for c0 in range(0, len(Rs), chunk):
        Rc = Rs[c0:c0 + chunk]
        xr = torch.einsum("mij,nj->mni", Rc, xyz)
        nr = torch.einsum("mij,nj->mni", Rc, nrm)
        feat = torch.cat([xr, nr], -1).transpose(1, 2).contiguous()
        lg = bb(feat, clsv.expand(len(Rc)))
        p = lg.exp()
        confs.append(p[:, :, PART_RANGES[cls]].max(-1).values.mean(-1))
        for m in range(len(Rc)):
            mious.append(shape_miou(mask_argmax(lg[m], cls), gt, cls))
    return np.array(mious), torch.cat(confs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corrupt", choices=["yaw", "tilt", "so3"], required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--dump", default="")
    args = ap.parse_args()

    bb = FrozenPartSeg(ckpt=args.ckpt)
    ds = ShapeNetPart(args.root, "test", npoints=None)
    idx = np.random.default_rng(0).choice(len(ds), args.n, replace=False).tolist()

    res = {k: [] for k in ["clean", "corrupt", "recovered", "oracle"]}
    for j, i in enumerate(idx):
        s = ds[i]
        xyz, nrm = s["points"].cuda(), s["normals"].cuda()
        gt, cls = s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        feat = torch.cat([xyz, nrm], -1).t().unsqueeze(0).contiguous()
        lg = bb(feat, clsv)
        res["clean"].append(shape_miou(mask_argmax(lg.squeeze(0), cls), gt, cls))

        crng = np.random.default_rng(5_000_000 + i)
        Rc = rand_rot(1, args.corrupt, "cuda", crng)[0]
        xc, nc = xyz @ Rc.T, nrm @ Rc.T
        Rs, ang = cand_pool(args.M, args.corrupt, "cuda", 200_000 + i)
        mious, conf = pool_eval(bb, xc, nc, cls, gt, Rs, ang)
        res["corrupt"].append(float(mious[0]))
        k = int((conf - LAM * ang).argmax().item())
        res["recovered"].append(float(mious[k]))
        res["oracle"].append(float(mious.max()))
        if (j + 1) % 32 == 0:
            print(f"  [{args.corrupt}] {j+1}/{len(idx)}", flush=True)

    m = {k: float(np.mean(v)) for k, v in res.items()}
    gap = m["clean"] - m["corrupt"] + 1e-9
    print(f"[partseg aug-ablation eval] ckpt={args.ckpt} corrupt={args.corrupt} n={len(idx)} M={args.M}")
    print(f"  clean={m['clean']:.3f} corrupt={m['corrupt']:.3f} recovered={m['recovered']:.3f} "
          f"oracle={m['oracle']:.3f}  (recovered {100*(m['recovered']-m['corrupt'])/gap:.0f}% of clean gap)")
    if args.dump:
        json.dump({"ckpt": args.ckpt, "corrupt": args.corrupt, "n": len(idx), **m}, open(args.dump, "w"))


if __name__ == "__main__":
    main()
