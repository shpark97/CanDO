"""Two more runnable baselines for the pose-robustness comparison (ShapeNet part-seg, SO(3)):

  DUA (Mirza et al., CVPR 2022): backprop-free test-time adaptation of BatchNorm running statistics from
      augmented copies of the test input. Recent, gradient-free, but adapts the model, not the input.
  Grad-canon: white-box gradient-based input canonicalization, gradient ascent on the rotation against the
      model's confidence (with restarts). The "if you had gradients" comparison; our multi-modal-landscape
      finding predicts it traps in local optima and loses to global gradient-free search.

Same per-index SO(3) corruption as baseline_compare_partseg, so the shapes match. Shardable.

Run:  PYTHONPATH=. python -m src.eval.baseline_tta_extra --n -1 --idx_start 0 --idx_end -1 --dump out.json
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.shapenet_part import ShapeNetPart
from src.eval.baseline_compare_partseg import NUM_CATEGORIES, build_trainable_partseg, miou_one
from src.downstream.pointnet2_partseg import FrozenPartSeg, make_input
from src.eval.metrics import mask_argmax, shape_miou
from src.zoo.canonicalize import Rso3


def dua_partseg(net, state, xyz, nrm, cls, gt, K=16, steps=3, momentum=0.1, seed=0):
    """DUA: adapt BN running stats from K augmented copies of the (corrupted) input, no backprop."""
    rng = np.random.default_rng(seed)
    oh1 = F.one_hot(torch.tensor([cls]), NUM_CATEGORIES).float().cuda()
    ohK = F.one_hot(torch.tensor([cls] * K), NUM_CATEGORIES).float().cuda()
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = momentum
    net.train()
    with torch.no_grad():
        for _ in range(steps):
            xs, ns = [], []
            for _ in range(K):
                s = torch.tensor(rng.uniform(0.9, 1.1, size=3), device=xyz.device, dtype=torch.float32)
                xs.append(xyz * s + 0.01 * torch.randn_like(xyz)); ns.append(F.normalize(nrm / s, dim=1))
            feat = torch.cat([torch.stack(xs), torch.stack(ns)], -1).transpose(1, 2).contiguous()
            net(feat, ohK)                                              # updates running stats only
    net.eval()
    with torch.no_grad():
        lg, _ = net(make_input(xyz, nrm, True), oh1)
        mi = shape_miou(mask_argmax(lg.squeeze(0), cls), gt, cls)
    net.load_state_dict(state)
    return mi


def gradcanon_partseg(net, xyz, nrm, cls, gt, restarts=3, steps=20, lr=0.2, seed=0):
    """White-box gradient ascent on the rotation maximizing confidence, with random restarts."""
    net.eval()
    oh1 = F.one_hot(torch.tensor([cls]), NUM_CATEGORIES).float().cuda()
    rng = np.random.default_rng(seed)
    best_conf, best_v = -1.0, torch.zeros(3, device=xyz.device)
    for r in range(restarts):
        init = np.zeros(3) if r == 0 else rng.normal(size=3) * 0.8
        v = torch.tensor(init, device=xyz.device, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.SGD([v], lr=lr, momentum=0.9)
        for t in range(steps):
            R = Rso3(v); lg, _ = net(make_input(xyz @ R.T, nrm @ R.T, True), oh1)
            loss = -lg.exp().max(-1).values.mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            R = Rso3(v); lg, _ = net(make_input(xyz @ R.T, nrm @ R.T, True), oh1)
            c = lg.exp().max(-1).values.mean().item()
        if c > best_conf:
            best_conf = c; best_v = v.detach().clone()
    with torch.no_grad():
        R = Rso3(best_v)
        lg, _ = net(make_input(xyz @ R.T, nrm @ R.T, True), oh1)
        return shape_miou(mask_argmax(lg.squeeze(0), cls), gt, cls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--idx_start", type=int, default=0); ap.add_argument("--idx_end", type=int, default=-1)
    ap.add_argument("--dump", type=str, default="")
    args = ap.parse_args()
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg()
    net = build_trainable_partseg(); state = copy.deepcopy(net.state_dict())
    idx = list(range(len(ds)))
    end = len(idx) if args.idx_end < 0 else min(args.idx_end, len(idx))
    shard = idx[args.idx_start:end]

    res = {k: [] for k in ["clean", "corrupt", "dua", "gradcanon"]}
    for i in shard:
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        res["clean"].append(miou_one(bb, xyz, nrm, clsv, cls, gt))
        crng = np.random.default_rng(7_000_000 + i)
        axis = crng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * crng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        R0 = Rso3(v0); xr, nr = xyz @ R0.T, nrm @ R0.T
        res["corrupt"].append(miou_one(bb, xr, nr, clsv, cls, gt))
        res["dua"].append(dua_partseg(net, state, xr, nr, cls, gt, seed=1_000_000 + i))
        res["gradcanon"].append(gradcanon_partseg(net, xr, nr, cls, gt, seed=2_000_000 + i))

    if args.dump:
        json.dump({"idx": shard, **res}, open(args.dump, "w"))
    m = {k: float(np.mean(v)) for k, v in res.items()}; gap = m["clean"] - m["corrupt"] + 1e-9
    print(f"[TTA-extra @ ShapeNet part-seg SO(3), n={len(shard)} (idx {args.idx_start}:{end})]")
    for k in ["corrupt", "dua", "gradcanon"]:
        print(f"  {k:10s} = {m[k]:.3f}   ({100*(m[k]-m['corrupt'])/gap:+.0f}% gap)")


if __name__ == "__main__":
    main()
