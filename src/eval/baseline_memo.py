"""Competitive baseline: MEMO (test-time param adaptation by marginal-entropy minimization) vs our
test-time POSE canonicalization, on ModelNet40 cls under SO(3) corruption.

MEMO (Zhang et al., NeurIPS'22) adapts the model's parameters (episodic) to minimize the entropy of the
prediction averaged over label-preserving augmentations (jitter+scale). It does NOT recover orientation,
so on a rotation-collapsed input it should barely help — whereas our gradient-free, label-free pose
search directly undoes the rotation. The comparison shows that pose corruption needs pose recovery;
param-adaptation TTA cannot do it (and needs gradients, which we do not).

Run:  python -m src.eval.baseline_memo --n 24
"""
from __future__ import annotations

import argparse
import copy
import os
import sys

import numpy as np
import torch

from src.data.modelnet import ModelNet40Cls
from src.downstream.modelnet_cls import FrozenModelNetCls, _CKPT, _PN2_ROOT
from src.eval.canon_modelnet import pred_at
from src.zoo.canonicalize import Rso3


def recover_cma_batched(bb, xc, budget=240, popsize=12, restarts=2, sigma0=1.2, seed=0):
    """CMA-ES over SO(3) axis-angle, label-free max-softmax confidence, with the population evaluated in
    ONE batched forward per generation (strong like CMA, fast like batched). Returns best axis-angle v."""
    import cma
    best_v, best_c = torch.zeros(3, device=xc.device), -1.0
    for r in range(restarts):
        x0 = (np.random.default_rng(seed + r).normal(size=3) * 0.3).tolist()
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"maxfevals": budget // restarts,
                                                   "popsize": popsize, "verbose": -9, "seed": seed + r + 1})
        while not es.stop():
            sols = es.ask()
            Rs = torch.stack([Rso3(torch.tensor(s, device=xc.device, dtype=torch.float32)) for s in sols])
            batch = torch.einsum("mij,nj->mni", Rs, xc)              # (P,N,3)
            conf = bb(batch).exp().max(-1).values                    # (P,) max-softmax
            es.tell(sols, [-c for c in conf.tolist()])
        v = torch.tensor(es.result.xbest, device=xc.device, dtype=torch.float32)
        c = bb((xc @ Rso3(v).T).unsqueeze(0)).exp().max().item()
        if c > best_c:
            best_c, best_v = c, v
    return best_v


def build_trainable_cls(device="cuda"):
    for p in (os.path.abspath(_PN2_ROOT), os.path.abspath(os.path.join(_PN2_ROOT, "models"))):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.pointnet2_cls_ssg import get_model
    net = get_model(40, normal_channel=False)
    net.load_state_dict(torch.load(_CKPT, map_location="cpu", weights_only=False)["model_state_dict"])
    return net.to(device)


def augment(xyz, rng):
    """label-preserving, N-preserving: jitter + anisotropic scale."""
    s = torch.tensor(rng.uniform(0.9, 1.1, size=3), device=xyz.device, dtype=torch.float32)
    return xyz * s + 0.01 * torch.randn_like(xyz)


def tent_predict(net, state, xyz, lr=1e-3, steps=1):
    """Episodic TENT: adapt parameters by minimizing the entropy of the prediction on the (corrupted)
    input itself, then predict and reset. Unlike MEMO it uses no augmentation. The model is kept in eval
    mode (running batch-norm statistics) so a single-sample forward is well defined; only the entropy
    gradient updates the parameters."""
    net.eval()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    x = xyz.transpose(0, 1).unsqueeze(0)                       # (1,3,N)
    for _ in range(steps):
        logits, _ = net(x)                                     # (1,40) log-prob
        p = logits.exp()
        loss = -(p * (p + 1e-12).log()).sum(-1).mean()         # entropy
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        out, _ = net(xyz.transpose(0, 1).unsqueeze(0))
    net.load_state_dict(state)
    return out.squeeze(0)


def memo_predict(net, state, xyz, K=16, lr=1e-3, seed=0):
    """Episodic MEMO: adapt params by 1 step of marginal-entropy min over K augs, predict, reset."""
    rng = np.random.default_rng(seed)
    net.train()
    opt = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    batch = torch.stack([augment(xyz, rng) for _ in range(K)]).transpose(1, 2).contiguous()  # (K,3,N)
    logits, _ = net(batch)                       # (K,40) log-prob
    p = logits.exp().mean(0)                      # marginal (40,)
    loss = -(p * (p + 1e-12).log()).sum()         # minimize marginal entropy
    opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        out, _ = net(xyz.transpose(0, 1).unsqueeze(0))  # (1,40)
    net.load_state_dict(state)                     # reset (episodic)
    return out.squeeze(0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = ModelNet40Cls("datasets/modelnet_dl", "test", 1024)
    bb = FrozenModelNetCls()                       # for source + ours (no-grad)
    memo_net = build_trainable_cls()               # separate trainable copy for MEMO
    memo_state = copy.deepcopy(memo_net.state_dict())
    rng = np.random.default_rng(args.seed); Z = torch.zeros(3, device="cuda")
    idx = list(range(len(ds))) if args.n < 0 else rng.choice(len(ds), args.n, replace=False).tolist()

    res = {"clean": [], "corrupt": [], "tent": [], "memo": [], "ours": []}
    for j, i in enumerate(idx):
        s = ds[i]; xyz = s["points"].cuda(); y = s["label"]
        res["clean"].append(int(pred_at(bb, xyz, Z).argmax().item() == y))
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * rng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        xr = xyz @ Rso3(v0).T
        res["corrupt"].append(int(pred_at(bb, xr, Z).argmax().item() == y))
        res["tent"].append(int(tent_predict(memo_net, memo_state, xr).argmax().item() == y))
        res["memo"].append(int(memo_predict(memo_net, memo_state, xr, seed=100 + j).argmax().item() == y))
        v = recover_cma_batched(bb, xr, budget=240, restarts=2, seed=200 + j)  # our method for SO(3)
        res["ours"].append(int(pred_at(bb, xr, v).argmax().item() == y))
        if args.n < 0 and (j + 1) % 200 == 0:
            print(f"  ... {j+1}/{len(idx)}", flush=True)

    m = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"[TTA baselines vs ours @ ModelNet40 cls SO(3), n={len(idx)}]")
    print(f"  clean (aligned)        = {m['clean']:.3f}")
    print(f"  corrupted (SO3)        = {m['corrupt']:.3f}")
    print(f"  TENT (backprop TTA)    = {m['tent']:.3f}   (Δ {m['tent']-m['corrupt']:+.3f})")
    print(f"  MEMO (backprop TTA)    = {m['memo']:.3f}   (Δ {m['memo']-m['corrupt']:+.3f})")
    print(f"  OURS (gradient-free)   = {m['ours']:.3f}   (Δ {m['ours']-m['corrupt']:+.3f})")


if __name__ == "__main__":
    main()
