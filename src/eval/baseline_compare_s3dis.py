"""Competitive baseline table on scene scale: S3DIS sem-seg under out-of-plane TILT corruption.

The scene analog of `baseline_compare_partseg` (which runs on ShapeNet part-seg under SO(3)). The frozen
yanx27 sem-seg model is z-rotation augmented (yaw-robust) but assumes gravity / z-up, so a pitch/roll tilt
collapses it. We compare, on the SAME tilted blocks:
  source / PCA-canon / PCA+conf / DUA / TENT / MEMO / white-box gradient-canon / ours (CMA-ES) / oracle.

Only ours and the white-box gradient baseline actually correct the input orientation; the gradient-free
adaptation baselines (PCA, DUA) and the parameter-space TTA baselines (TENT, MEMO) cannot, exactly as on
ShapeNet. Corruption here is a horizontal-axis tilt (not full SO(3)) because indoor scenes are gravity-
aligned, matching the S3DIS row of Table 1.

Per-block randomness (block choice, corruption, optimizer seeds) is keyed by the GLOBAL candidate index i,
so any contiguous shard [idx_start:idx_end] reproduces the single-process numbers exactly -> 8-GPU safe.

Run (single process):  PYTHONPATH=. python -m src.eval.baseline_compare_s3dis --n 24
Run (sharded):         ... --n 300 --idx_start A --idx_end B --dump runs/s3dis_cmpshard_g.json
Aggregate:             PYTHONPATH=. python -m src.eval.aggregate_compare_s3dis runs/s3dis_cmpshard_*.json
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

from src.data.s3dis import NUM_CLASSES, S3DISRooms, sample_block
from src.downstream.pointnet2_semseg import _DEFAULT_CKPT, _PN2_ROOT, FrozenSemSeg
from src.eval.canon_s3dis import block_miou
from src.samplers.classic import fps_sample
from src.zoo.canonicalize import Rso3

BLOCK_SEED = 5_000_000      # room pick per candidate index
GEN_SEED = 6_000_000        # block-sampling generator per candidate index
CORR_SEED = 7_000_000       # tilt corruption per candidate index
OURS_SEED = 2_000_000       # CMA-ES / grad restart seeds per candidate index
MEMO_SEED = 1_000_000       # augmentation seeds per candidate index


def horiz(v2, device="cuda"):
    """2-vector (a,b) -> axis-angle 3-vector (a,b,0): a tilt about a horizontal axis (yaw is free)."""
    return torch.tensor([float(v2[0]), float(v2[1]), 0.0], device=device, dtype=torch.float32)


def _columns(d, color, c, rmax):
    """Build the (K,9) yanx27 sem-seg feature from local (centered+rotated) coords d. Differentiable in d."""
    ac = d + c                                              # absolute coords (height + room-norm channels)
    return torch.stack([
        d[:, 0], d[:, 1], ac[:, 2],
        color[:, 0] / 255.0, color[:, 1] / 255.0, color[:, 2] / 255.0,
        ac[:, 0] / (rmax[0] + 1e-8), ac[:, 1] / (rmax[1] + 1e-8), ac[:, 2] / (rmax[2] + 1e-8),
    ], dim=1)


def _center(block, sel):
    coord = block["coord"][sel]
    c = block["center"].clone(); c[2] = coord[:, 2].mean()
    return coord, c, block["color"][sel], block["room_max"]


def input_from_local(block, sel, d):
    """(1,9,K) input from local coords d (already centered about c and rotated)."""
    _, c, color, rmax = _center(block, sel)
    return _columns(d, color, c, rmax).t().unsqueeze(0).contiguous()


def rot_input(block, sel, R):
    """(1,9,K) input with the block rotated by matrix R about its center."""
    coord, c, _, _ = _center(block, sel)
    return input_from_local(block, sel, (coord - c) @ R.T)


def local_corrupt(block, sel, v0):
    """Local coords after the tilt corruption v0 (the geometry every baseline starts from)."""
    coord, c, _, _ = _center(block, sel)
    return (coord - c) @ Rso3(v0).T


# ----------------------------------------------------------------------------- objectives
def conf_at(bb, block, sel, v):
    return bb(rot_input(block, sel, Rso3(v))).squeeze(0).exp().max(-1).values.mean().item()


def miou_at(bb, block, sel, gt, v):
    return block_miou(bb(rot_input(block, sel, Rso3(v))).squeeze(0).argmax(-1), gt)


def miou_local(bb, block, sel, gt, d):
    return block_miou(bb(input_from_local(block, sel, d)).squeeze(0).argmax(-1), gt)


# ----------------------------------------------------------------------------- gradient-free baselines
def pca_canon(bb, block, sel, gt, dc):
    """Classical principal-axis alignment of the tilted block (sign by per-axis skewness)."""
    m = dc.mean(0, keepdim=True); cc = dc - m
    _, V = torch.linalg.eigh(cc.T @ cc / cc.shape[0]); V = V[:, [2, 1, 0]]
    sign = torch.sign(((cc @ V) ** 3).sum(0)); sign[sign == 0] = 1.0; V = V * sign
    return miou_local(bb, block, sel, gt, cc @ V)


def pca_conf(bb, block, sel, gt, dc):
    """PCA frame with the sign/axis ambiguity resolved by the model's own confidence (stronger PCA)."""
    import itertools
    m = dc.mean(0, keepdim=True); cc = dc - m
    _, V = torch.linalg.eigh(cc.T @ cc / cc.shape[0]); V = V[:, [2, 1, 0]]
    cands = []
    for s in itertools.product([1.0, -1.0], repeat=3):
        Vs = V * torch.tensor(s, device=dc.device, dtype=torch.float32)
        if torch.det(Vs) < 0:
            continue                                        # proper rotations only
        cands.append(cc @ Vs)
    _, c, color, rmax = _center(block, sel)
    feat = torch.stack([_columns(d, color, c, rmax) for d in cands]).transpose(1, 2).contiguous()
    lg = bb(feat)                                           # (4,N,13) log-prob
    k = int(lg.exp().max(-1).values.mean(-1).argmax().item())
    return block_miou(lg[k].argmax(-1), gt)


# ----------------------------------------------------------------------------- trainable-net TTA baselines
def build_trainable_semseg(device="cuda"):
    if os.path.abspath(_PN2_ROOT) not in sys.path:
        sys.path.insert(0, os.path.abspath(_PN2_ROOT))
    from models.pointnet2_sem_seg import get_model
    net = get_model(NUM_CLASSES)
    net.load_state_dict(torch.load(_DEFAULT_CKPT, map_location="cpu", weights_only=False)["model_state_dict"])
    return net.to(device)


def _augment(dc, K, seed):
    """K scale+jitter augmentations of the corrupted local coords (pose-preserving, MEMO/DUA style)."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(K):
        s = float(rng.uniform(0.9, 1.1))
        out.append(dc * s + 0.01 * torch.randn_like(dc))
    return out


def dua_semseg(net, state, block, sel, dc, gt, feat0, K=16, seed=0):
    """DUA: adapt BN running statistics on augmented corrupted copies (no gradients), then predict."""
    _, c, color, rmax = _center(block, sel)
    net.train()
    with torch.no_grad():
        aug = _augment(dc, K, seed)
        feat = torch.stack([_columns(d, color, c, rmax) for d in aug]).transpose(1, 2).contiguous()
        net(feat)                                           # one train-mode pass updates BN running stats
    net.eval()
    with torch.no_grad():
        mi = block_miou(net(feat0)[0].squeeze(0).argmax(-1), gt)
    net.load_state_dict(state)
    return mi


def tent_semseg(net, state, block, sel, gt, feat0, lr=5e-4):
    """Episodic TENT: one step minimizing per-point prediction entropy of the corrupted input, then reset."""
    net.train(); opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    logits, _ = net(feat0)                                  # (1,N,13) log-prob
    p = logits.exp()[0]
    loss = -(p * logits[0]).sum(-1).mean()                  # entropy = -sum p*log p (logits are log-softmax)
    opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        mi = block_miou(net(feat0)[0].squeeze(0).argmax(-1), gt)
    net.load_state_dict(state)
    return mi


def memo_semseg(net, state, block, sel, dc, gt, feat0, K=8, lr=5e-4, seed=0):
    """Episodic MEMO: one step minimizing marginal entropy over K augmentations, then reset."""
    _, c, color, rmax = _center(block, sel)
    net.train(); opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    aug = _augment(dc, K, seed)
    feat = torch.stack([_columns(d, color, c, rmax) for d in aug]).transpose(1, 2).contiguous()
    logits, _ = net(feat)                                   # (K,N,13) log-prob
    p = logits.exp().mean(0)                                # marginal per-point (N,13)
    p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
    loss = -(p * (p + 1e-12).log()).sum(-1).mean()          # marginal entropy
    opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        mi = block_miou(net(feat0)[0].squeeze(0).argmax(-1), gt)
    net.load_state_dict(state)
    return mi


# ----------------------------------------------------------------------------- white-box gradient canon
def gradcanon_semseg(net, block, sel, v0, gt, budget=160, restarts=2, lr=0.05, seed=0):
    """White-box gradient ascent on max-softmax confidence over the 2-D tilt (exact input gradients)."""
    net.eval()
    coord, c, color, rmax = _center(block, sel)
    base = (coord - c)
    steps = max(1, budget // restarts)
    best_w, best_c = torch.zeros(2, device=base.device), -1.0
    for r in range(restarts):
        w = torch.tensor(np.random.default_rng(seed + r).normal(size=2) * 0.2,
                         device=base.device, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([w], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            v = v0 + torch.stack([w[0], w[1], torch.zeros((), device=base.device)])
            d = base @ Rso3(v).T
            logits, _ = net(_columns(d, color, c, rmax).t().unsqueeze(0).contiguous())
            (-logits.exp().max(-1).values.mean()).backward()
            opt.step()
        with torch.no_grad():
            cc = conf_at_local(net, base, color, c, rmax, v0, w.detach())
        if cc > best_c:
            best_c, best_w = cc, w.detach().clone()
    return v0 + horiz(best_w.tolist(), device=base.device)


def conf_at_local(net, base, color, c, rmax, v0, w):
    v = v0 + torch.stack([w[0], w[1], torch.zeros((), device=base.device)])
    d = base @ Rso3(v).T
    logits, _ = net(_columns(d, color, c, rmax).t().unsqueeze(0).contiguous())
    return logits.exp().max(-1).values.mean().item()


# ----------------------------------------------------------------------------- ours (CMA-ES, black-box)
def ours_cma(bb, block, sel, v0, budget=160, sigma0=0.5, restarts=2, popsize=8, seed=0):
    import cma
    best_w, best_c = torch.zeros(2), -1.0
    for r in range(restarts):
        x0 = (np.random.default_rng(seed + r).normal(size=2) * 0.2).tolist()
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"maxfevals": budget // restarts,
                                                   "popsize": popsize, "verbose": -9, "seed": seed + r + 1})
        while not es.stop():
            sols = es.ask()
            es.tell(sols, [-conf_at(bb, block, sel, v0 + horiz(x)) for x in sols])
        c = conf_at(bb, block, sel, v0 + horiz(es.result.xbest))
        if c > best_c:
            best_c, best_w = c, torch.tensor(es.result.xbest)
    return v0 + horiz(best_w.tolist())


# ----------------------------------------------------------------------------- deterministic block list
def make_block(ds, cache, i, npoints):
    """Deterministic block for candidate index i (independent of any other index -> shard-safe)."""
    room_idx = int(np.random.default_rng(BLOCK_SEED + i).integers(len(ds)))
    if room_idx not in cache:
        cache[room_idx] = ds.load(room_idx)
    g = torch.Generator(device="cuda"); g.manual_seed(GEN_SEED + i)
    block = sample_block(cache[room_idx], 1.0, npoints, generator=g)
    if block is None:
        return None
    sel = fps_sample(block["coord"], npoints); gt = block["seg"][sel]
    if torch.unique(gt).numel() < 2:
        return None
    return block, sel, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/s3dis")
    ap.add_argument("--n", type=int, default=24, help="number of candidate indices to enumerate")
    ap.add_argument("--npoints", type=int, default=4096)
    ap.add_argument("--idx_start", type=int, default=0)
    ap.add_argument("--idx_end", type=int, default=-1, help="process candidate idx[idx_start:idx_end]; -1 = end")
    ap.add_argument("--dump", type=str, default="", help="write per-block raw results to this JSON path")
    ap.add_argument("--fast", action="store_true", help="skip pca/pcaconf/dua/tent/memo/grad (clean/corrupt/ours/oracle only)")
    args = ap.parse_args()

    ds = S3DISRooms(args.root, split="test", test_area=5)
    bb = FrozenSemSeg()
    gnet = build_trainable_semseg(); gstate = copy.deepcopy(gnet.state_dict())
    grid = np.linspace(-0.9, 0.9, 7)

    end = args.n if args.idx_end < 0 else min(args.idx_end, args.n)
    keys = ["clean", "corrupt", "pca", "pcaconf", "dua", "tent", "memo", "grad", "ours", "oracle"]
    res = {k: [] for k in keys}; done_idx = []
    cache = {}
    for i in range(args.idx_start, end):
        got = make_block(ds, cache, i, args.npoints)
        if got is None:
            continue
        block, sel, gt = got
        done_idx.append(i)
        # tilt corruption: horizontal axis, angle 0.3-0.9 rad (~17-52 deg)
        crng = np.random.default_rng(CORR_SEED + i)
        ang = crng.uniform(0.3, 0.9); phi = crng.uniform(0, 2 * np.pi)
        v0 = horiz([np.cos(phi) * ang, np.sin(phi) * ang])
        dc = local_corrupt(block, sel, v0)
        feat0 = rot_input(block, sel, Rso3(v0))

        res["clean"].append(miou_at(bb, block, sel, gt, torch.zeros(3, device="cuda")))
        res["corrupt"].append(block_miou(bb(feat0).squeeze(0).argmax(-1), gt))
        if not args.fast:
            res["pca"].append(pca_canon(bb, block, sel, gt, dc))
            res["pcaconf"].append(pca_conf(bb, block, sel, gt, dc))
            res["dua"].append(dua_semseg(gnet, gstate, block, sel, dc, gt, feat0, seed=MEMO_SEED + i))
            res["tent"].append(tent_semseg(gnet, gstate, block, sel, gt, feat0))
            res["memo"].append(memo_semseg(gnet, gstate, block, sel, dc, gt, feat0, seed=MEMO_SEED + i))
            vg = gradcanon_semseg(gnet, block, sel, v0, gt, seed=OURS_SEED + i)
            gnet.load_state_dict(gstate)                    # grad canon left BN in eval; keep weights pristine
            res["grad"].append(miou_at(bb, block, sel, gt, vg))
        vo = ours_cma(bb, block, sel, v0, seed=OURS_SEED + i)
        res["ours"].append(miou_at(bb, block, sel, gt, vo))
        # oracle: best residual tilt by GT over a 7x7 horizontal grid
        best = -1.0
        for a in grid:
            for b in grid:
                best = max(best, miou_at(bb, block, sel, gt, v0 + horiz([a, b])))
        res["oracle"].append(best)
        if len(done_idx) % 10 == 0:
            print(f"  ... {len(done_idx)} blocks (cand idx {i})", flush=True)

    if args.dump:
        import json
        json.dump({"idx": done_idx, **res}, open(args.dump, "w"))
        print(f"  dumped {len(done_idx)} blocks -> {args.dump}", flush=True)

    if not res["clean"]:
        print("no valid blocks in this shard"); return
    m = {k: float(np.mean(v)) for k, v in res.items() if v}; gap = m["clean"] - m["corrupt"]
    print(f"[Competitive @ S3DIS sem-seg TILT, n={len(done_idx)} blocks (cand {args.idx_start}:{end})]  "
          f"clean {m['clean']:.3f} corrupt {m['corrupt']:.3f} gap {gap:.3f}")
    for k in keys[1:]:
        if k in m:
            pct = 100 * (m[k] - m["corrupt"]) / (gap + 1e-9)
            print(f"  {k:8s} = {m[k]:.3f}   ({pct:+.0f}% gap)")


if __name__ == "__main__":
    main()
