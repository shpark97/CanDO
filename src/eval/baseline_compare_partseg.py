"""Table 1 and Table 2, ShapeNetPart under SO(3) corruption: source / PCA-canon / PCA+conf / TENT / MEMO /
ours (CMA-ES with batched populations) / oracle, all on the same shapes.

- ours: CMA-ES over axis-angle, label-free max-confidence, with the population evaluated in ONE batched
  forward per generation.
- MEMO(part-seg): episodic param adaptation by marginal-entropy min over jitter+scale augmentations
  (cannot recover pose).
- PCA-canon: classical principal-axis alignment (sign by skewness).

Run:  python -m src.eval.baseline_compare_partseg --n 48
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg, NUM_CATEGORIES, _PN2_ROOT, make_input
from src.eval.metrics import PART_RANGES, mask_argmax, shape_miou
from src.zoo.canonicalize import Rso3


def miou_one(bb, xyz, nrm, clsv, cls, gt):
    lg = bb(make_input(xyz, nrm, True), clsv).squeeze(0)
    return shape_miou(mask_argmax(lg, cls), gt, cls)


def batched_conf(bb, xyz, nrm, cls, vs):
    Rs = torch.stack([Rso3(v) for v in vs])
    xr = torch.einsum("mij,nj->mni", Rs, xyz); nr = torch.einsum("mij,nj->mni", Rs, nrm)
    feat = torch.cat([xr, nr], -1).transpose(1, 2).contiguous()
    lg = bb(feat, torch.full((len(vs),), cls, dtype=torch.long))     # (M,N,50)
    conf = lg.exp()[:, :, PART_RANGES[cls]].max(-1).values.mean(-1)  # (M,)
    return lg, conf


def ours_cma(bb, xyz, nrm, cls, budget=300, popsize=12, restarts=2, sigma0=1.0, seed=0):
    import cma
    best_v, best_c = torch.zeros(3, device=xyz.device), -1.0
    for r in range(restarts):
        x0 = (np.random.default_rng(seed + r).normal(size=3) * 0.3).tolist()
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"maxfevals": budget // restarts,
                                                   "popsize": popsize, "verbose": -9, "seed": seed + r + 1})
        while not es.stop():
            sols = es.ask()
            vs = [torch.tensor(s, device=xyz.device, dtype=torch.float32) for s in sols]
            _, conf = batched_conf(bb, xyz, nrm, cls, vs)            # batched population eval
            es.tell(sols, [-c for c in conf.tolist()])
        v = torch.tensor(es.result.xbest, device=xyz.device, dtype=torch.float32)
        c = batched_conf(bb, xyz, nrm, cls, [v])[1].item()
        if c > best_c:
            best_c, best_v = c, v
    return best_v


def pca_canon(xyz, nrm):
    c = xyz - xyz.mean(0, keepdim=True)
    _, V = torch.linalg.eigh(c.T @ c / c.shape[0]); V = V[:, [2, 1, 0]]
    sign = torch.sign(((c @ V) ** 3).sum(0)); sign[sign == 0] = 1.0; V = V * sign
    return c @ V, nrm @ V


def pca_conf(bb, xyz, nrm, cls, gt):
    """PCA frame with sign ambiguity resolved by the model's confidence (stronger PCA baseline):
    enumerate the four proper-rotation sign flips of the principal axes and keep the most confident."""
    import itertools
    c = xyz - xyz.mean(0, keepdim=True)
    _, V = torch.linalg.eigh(c.T @ c / c.shape[0]); V = V[:, [2, 1, 0]]
    xs, ns = [], []
    for s in itertools.product([1.0, -1.0], repeat=3):
        Vs = V * torch.tensor(s, device=xyz.device, dtype=torch.float32)
        if torch.det(Vs) < 0:
            continue                                    # keep proper rotations only
        xs.append(c @ Vs); ns.append(nrm @ Vs)
    feat = torch.cat([torch.stack(xs), torch.stack(ns)], -1).transpose(1, 2).contiguous()
    lg = bb(feat, torch.full((len(xs),), cls, dtype=torch.long))
    conf = lg.exp()[:, :, PART_RANGES[cls]].max(-1).values.mean(-1)
    k = int(conf.argmax().item())
    return shape_miou(mask_argmax(lg[k], cls), gt, cls)


def build_trainable_partseg(device="cuda"):
    if os.path.abspath(_PN2_ROOT) not in sys.path:
        sys.path.insert(0, os.path.abspath(_PN2_ROOT))
    from models.pointnet2_part_seg_msg import get_model
    net = get_model(50, normal_channel=True)
    ck = os.path.join(_PN2_ROOT, "log", "part_seg", "pointnet2_part_seg_msg", "checkpoints", "best_model.pth")
    net.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["model_state_dict"])
    return net.to(device)


def memo_partseg(net, state, xyz, nrm, cls, gt, K=8, lr=5e-4, seed=0):
    import torch.nn.functional as F
    rng = np.random.default_rng(seed); parts = PART_RANGES[cls]
    onehot = F.one_hot(torch.tensor([cls] * K), NUM_CATEGORIES).float().cuda()
    net.train(); opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    xs, ns = [], []
    for _ in range(K):
        s = torch.tensor(rng.uniform(0.9, 1.1, size=3), device=xyz.device, dtype=torch.float32)
        xs.append(xyz * s + 0.01 * torch.randn_like(xyz))
        ns.append(F.normalize(nrm / s, dim=1))
    feat = torch.cat([torch.stack(xs), torch.stack(ns)], -1).transpose(1, 2).contiguous()  # (K,6,N)
    logits, _ = net(feat, onehot)                                   # (K,N,50) log-prob
    p = logits.exp()[:, :, parts].mean(0)                            # marginal per-point (N,|parts|)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
    loss = -(p * (p + 1e-12).log()).sum(-1).mean()                   # marginal entropy
    opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        oh1 = F.one_hot(torch.tensor([cls]), NUM_CATEGORIES).float().cuda()
        lg, _ = net(make_input(xyz, nrm, True), oh1)
        mi = shape_miou(mask_argmax(lg.squeeze(0), cls), gt, cls)
    net.load_state_dict(state)
    return mi


def tent_partseg(net, state, xyz, nrm, cls, gt, lr=5e-4):
    """Episodic TENT on segmentation: one step minimizing the per-point prediction entropy of the
    (corrupted) input, then predict and reset. Unlike MEMO it uses no augmentation."""
    import torch.nn.functional as F
    parts = PART_RANGES[cls]
    oh1 = F.one_hot(torch.tensor([cls]), NUM_CATEGORIES).float().cuda()
    net.train(); opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    logits, _ = net(make_input(xyz, nrm, True), oh1)                  # (1,N,50)
    p = logits.exp()[0][:, parts]
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
    loss = -(p * (p + 1e-12).log()).sum(-1).mean()                   # per-point entropy
    opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        lg, _ = net(make_input(xyz, nrm, True), oh1)
        mi = shape_miou(mask_argmax(lg.squeeze(0), cls), gt, cls)
    net.load_state_dict(state)
    return mi


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--Moracle", type=int, default=256)
    ap.add_argument("--idx_start", type=int, default=0)
    ap.add_argument("--idx_end", type=int, default=-1, help="process idx[idx_start:idx_end]; -1 = end")
    ap.add_argument("--dump", type=str, default="", help="write per-shape raw results to this JSON path")
    ap.add_argument("--cseed", type=int, default=0, help="corruption-realization seed offset (0 = headline run)")
    ap.add_argument("--fast", action="store_true", help="skip pca/pcaconf/tent/memo (only clean/corrupt/ours/oracle)")
    args = ap.parse_args()
    OFF = args.cseed * 100_000_000                                   # distinct corruption draw per seed
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg()
    memo_net = build_trainable_partseg(); memo_state = copy.deepcopy(memo_net.state_dict())
    rng = np.random.default_rng(args.seed)
    idx = list(range(len(ds))) if args.n < 0 else rng.choice(len(ds), args.n, replace=False).tolist()
    end = len(idx) if args.idx_end < 0 else min(args.idx_end, len(idx))
    shard = idx[args.idx_start:end]
    # All per-shape randomness is keyed by the GLOBAL index i (not processing order), so any contiguous
    # shard reproduces exactly the same numbers as the full single-process run -> safe to split across GPUs.

    res = {k: [] for k in ["clean", "corrupt", "pca", "pcaconf", "tent", "memo", "ours", "oracle"]}
    for j, i in enumerate(shard):
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        res["clean"].append(miou_one(bb, xyz, nrm, clsv, cls, gt))
        crng0 = np.random.default_rng(7_000_000 + i + OFF)         # per-index corruption (shard-invariant)
        axis = crng0.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * crng0.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        R0 = Rso3(v0); xr, nr = xyz @ R0.T, nrm @ R0.T
        res["corrupt"].append(miou_one(bb, xr, nr, clsv, cls, gt))
        if not args.fast:
            xp, npc = pca_canon(xr, nr); res["pca"].append(miou_one(bb, xp, npc, clsv, cls, gt))
            res["pcaconf"].append(pca_conf(bb, xr, nr, cls, gt))
            res["tent"].append(tent_partseg(memo_net, memo_state, xr, nr, cls, gt))
            res["memo"].append(memo_partseg(memo_net, memo_state, xr, nr, cls, gt, seed=1_000_000 + i + OFF))
        if (j + 1) % 100 == 0:
            print(f"  ... {j+1}/{len(shard)} (idx {i})", flush=True)
        v = ours_cma(bb, xr, nr, cls, seed=2_000_000 + i + OFF); R = Rso3(v)
        res["ours"].append(miou_one(bb, xr @ R.T, nr @ R.T, clsv, cls, gt))
        # oracle: max mIoU over Moracle random candidates (batched, chunked to bound peak memory)
        crng = np.random.default_rng(3_000_000 + i + OFF)
        vs = [torch.zeros(3, device="cuda")]
        for _ in range(args.Moracle - 1):
            a = crng.normal(size=3); a /= np.linalg.norm(a) + 1e-9
            vs.append(torch.tensor(a * crng.uniform(0, np.pi), device="cuda", dtype=torch.float32))
        omax = 0.0
        for c in range(0, len(vs), 64):
            lg, _ = batched_conf(bb, xr, nr, cls, vs[c:c + 64])
            omax = max(omax, max(shape_miou(mask_argmax(lg[m], cls), gt, cls) for m in range(lg.shape[0])))
        res["oracle"].append(omax)

    if args.dump:
        import json
        json.dump({"idx": shard, **res}, open(args.dump, "w"))
        print(f"  dumped {len(shard)} shapes -> {args.dump}", flush=True)

    m = {k: float(np.mean(v)) for k, v in res.items()}; gap = m["clean"] - m["corrupt"]
    print(f"[Competitive @ ShapeNet part-seg SO(3), n={len(shard)} (idx {args.idx_start}:{end})]  "
          f"clean {m['clean']:.3f} corrupt {m['corrupt']:.3f} gap {gap:.3f}")
    for k in ["corrupt", "pca", "pcaconf", "tent", "memo", "ours", "oracle"]:
        pct = 100 * (m[k] - m["corrupt"]) / (gap + 1e-9)
        print(f"  {k:8s} = {m[k]:.3f}   ({pct:+.0f}% gap)")


if __name__ == "__main__":
    main()
