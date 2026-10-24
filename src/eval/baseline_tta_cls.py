"""cls Acc cells for DUA and Gradient-Based Canonicalization (ModelNet40, SO(3)).

  DUA: backprop-free BatchNorm-statistic adaptation from augmented copies of the (corrupted) input.
  Grad-canon: white-box gradient ascent on the rotation maximizing the classifier's confidence.

Same per-index SO(3) corruption as baseline_cls_extra, so the shapes match. Shardable.

Run:  PYTHONPATH=. python -m src.eval.baseline_tta_cls --n -1 --idx_start 0 --idx_end -1 --dump out.json
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch
import torch.nn as nn

from src.data.modelnet import ModelNet40Cls
from src.downstream.modelnet_cls import FrozenModelNetCls
from src.eval.baseline_memo import augment, build_trainable_cls
from src.eval.canon_modelnet import pred_at
from src.zoo.canonicalize import Rso3


def dua_cls(net, state, xyz, y, K=16, steps=3, momentum=0.1, seed=0):
    rng = np.random.default_rng(seed)
    for m in net.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = momentum
    net.train()
    with torch.no_grad():
        for _ in range(steps):
            batch = torch.stack([augment(xyz, rng) for _ in range(K)]).transpose(1, 2).contiguous()  # (K,3,N)
            net(batch)
    net.eval()
    with torch.no_grad():
        out, _ = net(xyz.transpose(0, 1).unsqueeze(0))
    net.load_state_dict(state)
    return int(out.squeeze(0).argmax().item() == y)


def gradcanon_cls(net, xyz, y, restarts=3, steps=20, lr=0.2, seed=0):
    net.eval()
    rng = np.random.default_rng(seed)
    best_conf, best_v = -1.0, torch.zeros(3, device=xyz.device)
    for r in range(restarts):
        init = np.zeros(3) if r == 0 else rng.normal(size=3) * 0.8
        v = torch.tensor(init, device=xyz.device, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.SGD([v], lr=lr, momentum=0.9)
        for _ in range(steps):
            x = (xyz @ Rso3(v).T).transpose(0, 1).unsqueeze(0)
            logits, _ = net(x)
            loss = -logits.exp().max()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            logits, _ = net((xyz @ Rso3(v).T).transpose(0, 1).unsqueeze(0))
            c = logits.exp().max().item()
        if c > best_conf:
            best_conf = c; best_v = v.detach().clone()
    with torch.no_grad():
        logits, _ = net((xyz @ Rso3(best_v).T).transpose(0, 1).unsqueeze(0))
        return int(logits.squeeze(0).argmax().item() == y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/modelnet_dl"); ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--n", type=int, default=-1)
    ap.add_argument("--idx_start", type=int, default=0); ap.add_argument("--idx_end", type=int, default=-1)
    ap.add_argument("--dump", type=str, default="")
    args = ap.parse_args()
    ds = ModelNet40Cls(args.root, "test", args.npoints); bb = FrozenModelNetCls()
    net = build_trainable_cls(); state = copy.deepcopy(net.state_dict())
    idx = list(range(len(ds)))
    end = len(idx) if args.idx_end < 0 else min(args.idx_end, len(idx))
    shard = idx[args.idx_start:end]
    Z = torch.zeros(3, device="cuda")

    res = {k: [] for k in ["source", "dua", "gradcanon"]}
    for i in shard:
        s = ds[i]; xyz = s["points"].cuda().float(); y = s["label"]
        crng = np.random.default_rng(7_000_000 + i)
        axis = crng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        xc = xyz @ Rso3(torch.tensor(axis * crng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)).T
        res["source"].append(int(pred_at(bb, xc, Z).argmax().item() == y))
        res["dua"].append(dua_cls(net, state, xc, y, seed=1_000_000 + i))
        res["gradcanon"].append(gradcanon_cls(net, xc, y, seed=2_000_000 + i))

    if args.dump:
        json.dump({"idx": shard, **res}, open(args.dump, "w"))
    m = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"[TTA-cls @ ModelNet40 SO(3), n={len(shard)} (idx {args.idx_start}:{end})]")
    for k in ["source", "dua", "gradcanon"]:
        print(f"  {k:10s} acc = {m[k]:.3f}")


if __name__ == "__main__":
    main()
