"""Table 5: gradient-free optimizers under a matched query budget,
on the same canonicalization task -- ShapeNet SO(3) (3-D, the case where optimizer choice matters most).

Compares SPSA, NES, Coordinate-FD, CMA-ES, and random search at ~300 black-box queries/shape each.
Deterministic per-shape seeding, so it is reproducible.

Run:  python -m src.eval.estimator_ablation --n 24
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg, make_input
from src.eval.metrics import mask_argmax, shape_miou
from src.zoo.canonicalize import Rso3, confidence_so3, zoo_canonicalize_so3, zoo_canonicalize_so3_cma
from src.zoo.estimator import NES, REINFORCE, SPSA, Coordinate
from src.eval.bo_query_efficiency import bo_maximize

BUDGET = 300


def miou_with_v(bb, xr, nr, clsv, cls, gt, v):
    R = Rso3(v)
    lg = bb(make_input(xr @ R.T, nr @ R.T, True), clsv).squeeze(0)
    return shape_miou(mask_argmax(lg, cls), gt, cls)


def random_search(bb, xr, nr, clsv, n_queries, seed):
    rng = np.random.default_rng(seed)
    best_v, best_c = torch.zeros(3, device=xr.device), -1.0
    for _ in range(n_queries):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v = torch.tensor(axis * rng.uniform(0, np.pi), device=xr.device, dtype=torch.float32)
        c = confidence_so3(bb, xr, nr, clsv, v).item()
        if c > best_c:
            best_c, best_v = c, v
    return best_v, n_queries


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=24); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg(); rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), args.n, replace=False).tolist()

    # estimator -> recovery callable returning (v, nq); budgets tuned to ~BUDGET evals/shape
    def spsa(xr, nr, clsv, sd):
        return zoo_canonicalize_so3(bb, xr, nr, clsv, SPSA(c=0.1, avg=1), restarts=3, steps=49, seed=sd)

    def nes(xr, nr, clsv, sd):
        return zoo_canonicalize_so3(bb, xr, nr, clsv, NES(sigma=0.15, m=4), restarts=3, steps=12, seed=sd)

    def coord(xr, nr, clsv, sd):
        return zoo_canonicalize_so3(bb, xr, nr, clsv, Coordinate(eps=0.1), restarts=3, steps=16, seed=sd)

    def reinforce(xr, nr, clsv, sd):
        return zoo_canonicalize_so3(bb, xr, nr, clsv, REINFORCE(sigma=0.15, m=8), restarts=3, steps=12, seed=sd)

    def cma(xr, nr, clsv, sd):
        return zoo_canonicalize_so3_cma(bb, xr, nr, clsv, budget=BUDGET, restarts=2, seed=sd)

    def bo(xr, nr, clsv, sd):
        f = lambda x: confidence_so3(bb, xr, nr, clsv,  # noqa: E731
                                     torch.tensor(x, device="cuda", dtype=torch.float32)).item()
        vbo = bo_maximize(f, BUDGET, seed=sd)
        return torch.tensor(vbo, device="cuda", dtype=torch.float32), BUDGET

    methods = {"SPSA": spsa, "NES": nes, "Coordinate-FD": coord, "REINFORCE (policy-grad)": reinforce,
               "CMA-ES": cma, "Bayesian": bo}
    res = {"clean": [], "corrupt": []}
    rec = {m: [] for m in list(methods) + ["random"]}
    q = {m: [] for m in list(methods) + ["random"]}

    for j, i in enumerate(idx):
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        res["clean"].append(miou_with_v(bb, xyz, nrm, clsv, cls, gt, torch.zeros(3, device="cuda")))
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * rng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        R0 = Rso3(v0); xr, nr = xyz @ R0.T, nrm @ R0.T
        res["corrupt"].append(miou_with_v(bb, xr, nr, clsv, cls, gt, torch.zeros(3, device="cuda")))
        sd = 1000 + j  # deterministic per shape
        for m, fn in methods.items():
            v, nq = fn(xr, nr, clsv, sd)
            rec[m].append(miou_with_v(bb, xr, nr, clsv, cls, gt, v)); q[m].append(nq)
        vR, nqR = random_search(bb, xr, nr, clsv, BUDGET, sd)
        rec["random"].append(miou_with_v(bb, xr, nr, clsv, cls, gt, vR)); q["random"].append(nqR)

    cl, co = np.mean(res["clean"]), np.mean(res["corrupt"])
    print(f"[Estimator ablation @ ShapeNet SO(3), n={args.n}, matched budget ~{BUDGET} q/shape, seed {args.seed}]")
    print(f"  clean {cl:.3f} | corrupted {co:.3f} | gap {cl-co:.3f}\n")
    print(f"  {'estimator':14s} {'recovered':>10s} {'queries':>8s} {'%gap':>6s}")
    order = sorted(list(methods) + ["random"], key=lambda m: np.mean(rec[m]))
    for m in order:
        r = np.mean(rec[m]); pct = 100 * (r - co) / (cl - co + 1e-9)
        print(f"  {m:14s} {r:10.3f} {np.mean(q[m]):8.0f} {pct:5.0f}%")


if __name__ == "__main__":
    main()
