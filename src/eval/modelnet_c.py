"""ModelNet-C: standard corruption benchmark eval of our label-free gradient-free canonicalization.

For each of the 7 official atomic corruptions x 5 severities, compare:
  - source  : frozen cls, corrupted input, no adaptation
  - ours    : same, after label-free SO(3) canonicalization (CMA-ES on max-softmax confidence)
The headroom-boundary prediction: ours repairs ROTATE (pose), is ~neutral on scale, and does NOT help
density/noise (dropout/add/jitter) -- those need a generative prior (cf. CloudFixer), not a pose fix.

Run:  python -m src.eval.modelnet_c --n 32
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.corruptions import CORRUPTIONS, FAMILY, corrupt, fix_npoints
from src.data.modelnet import ModelNet40Cls
from src.downstream.modelnet_cls import FrozenModelNetCls
from src.eval.canon_modelnet import pred_at
from src.zoo.canonicalize import Rso3


def _candidates(xc, M, seed):
    rng = np.random.default_rng(seed)
    vs = [torch.zeros(3, device=xc.device)]; norms = [0.0]
    for _ in range(M - 1):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9; ang = rng.uniform(0, np.pi)
        vs.append(torch.tensor(axis * ang, device=xc.device, dtype=torch.float32)); norms.append(ang)
    Rs = torch.stack([Rso3(v) for v in vs])                       # (M,3,3)
    return vs, torch.tensor(norms, device=xc.device), Rs


def recover_batched(bb, xc, M=96, lam=0.08, seed=0):
    """SO(3) canonicalization via batched random search + IDENTITY PRIOR, objective = raw confidence.
    score = max-softmax - lam*||v||. Identity always a candidate. (Gameable on outlier corruptions.)"""
    vs, norms, Rs = _candidates(xc, M, seed)
    batch = torch.einsum("mij,nj->mni", Rs, xc)
    conf = bb(batch).exp().max(-1).values
    k = int((conf - lam * norms).argmax().item())
    return vs[k], k == 0


def recover_gated(bb, xc, M=96, lam=0.08, K=4, drop=0.1, seed=0):
    """NON-GAMEABLE objective (MEMO-style marginal entropy): for each candidate rotation, average the
    softmax over K random sub-samples (drop `drop` of points) and score by LOW marginal entropy — i.e.
    confident AND consistent across resamples. A spurious rotation that exploits outliers gives
    inconsistent predictions across resamples -> high marginal entropy -> rejected. Identity prior keeps
    non-mis-oriented inputs put. Returns (best_v, chose_identity)."""
    vs, norms, Rs = _candidates(xc, M, seed)
    base = torch.einsum("mij,nj->mni", Rs, xc)                    # (M,N,3) rotated clouds
    N = base.shape[1]; keep = max(int(N * (1 - drop)), 256)
    rng = np.random.default_rng(seed + 1)
    pbar = torch.zeros(len(vs), bb(base[:1]).shape[-1], device=xc.device)  # (M,C)
    for _ in range(K):
        idx = torch.tensor(rng.choice(N, keep, replace=False), device=xc.device)
        pbar += bb(base[:, idx, :]).exp()                         # (M,C) softmax of each resample
    pbar /= K
    ent = -(pbar * (pbar + 1e-12).log()).sum(-1)                  # (M,) marginal entropy
    score = -ent - lam * norms                                    # maximize: low entropy, small rotation
    k = int(score.argmax().item())
    return vs[k], k == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/modelnet_dl"); ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--levels", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--M", type=int, default=96); ap.add_argument("--lam", type=float, default=0.08)
    ap.add_argument("--gate", choices=["conf", "margent"], default="conf",
                    help="conf = raw max-softmax (gameable); margent = MEMO marginal-entropy (non-gameable)")
    ap.add_argument("--types", type=str, nargs="+", default=list(CORRUPTIONS.keys()))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = ModelNet40Cls(args.root, "test", 1024); bb = FrozenModelNetCls()
    recover = (lambda xc, sd: recover_gated(bb, xc, M=args.M, lam=args.lam, seed=sd)) if args.gate == "margent" \
        else (lambda xc, sd: recover_batched(bb, xc, M=args.M, lam=args.lam, seed=sd))
    rng = np.random.default_rng(args.seed)
    idx = list(range(len(ds))) if args.n < 0 else rng.choice(len(ds), args.n, replace=False).tolist()
    Z = torch.zeros(3, device="cuda")

    # cache clean normalized clouds + labels
    shapes = []
    clean_correct = 0
    for i in idx:
        s = ds[i]; xyz = s["points"].cuda(); y = s["label"]
        shapes.append((xyz.cpu().numpy().astype("float32"), y))
        clean_correct += int(pred_at(bb, xyz, Z).argmax().item() == y)
    print(f"[ModelNet-C @ frozen PointNet++ cls, n={len(idx)}, levels={args.levels}]  clean acc = {clean_correct/len(idx):.3f}\n")

    rows = []
    for ctype in args.types:
        src_acc, our_acc, id_frac = [], [], []
        for lvl in args.levels:
            for sj, (xyz_np, y) in enumerate(shapes):
                np.random.seed(1000 * lvl + sj)  # deterministic corruption draw
                cor = fix_npoints(corrupt(xyz_np, ctype, lvl), 1024)
                xc = torch.from_numpy(cor).cuda()
                src_acc.append(int(pred_at(bb, xc, Z).argmax().item() == y))
                v, chose_id = recover(xc, 7000 + sj)
                our_acc.append(int(pred_at(bb, xc, v).argmax().item() == y)); id_frac.append(int(chose_id))
        s, o = np.mean(src_acc), np.mean(our_acc)
        rows.append((ctype, FAMILY[ctype], s, o, o - s))
        print(f"  {ctype:16s} ({FAMILY[ctype]:9s}): source {s:.3f}  ours {o:.3f}  Δ {o-s:+.3f}  "
              f"(identity kept {np.mean(id_frac)*100:.0f}%)", flush=True)

    print("\n  family means:")
    for fam in ["transform", "noise", "density"]:
        r = [x for x in rows if x[1] == fam]
        ms, mo = np.mean([x[2] for x in r]), np.mean([x[3] for x in r])
        print(f"    {fam:9s}: source {ms:.3f}  ours {mo:.3f}  Δ {mo-ms:+.3f}")
    alls, allo = np.mean([x[2] for x in rows]), np.mean([x[3] for x in rows])
    print(f"  mean over 7 corruptions: source {alls:.3f}  ours {allo:.3f}  Δ {allo-alls:+.3f}")
    print("  (prediction: Δ>0 for rotate; ~0 for scale; ≤0/none for noise & density)")


if __name__ == "__main__":
    main()
