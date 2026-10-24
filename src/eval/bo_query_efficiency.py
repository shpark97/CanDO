"""Table 6: query efficiency of Bayesian optimization vs. random search and CMA-ES.

Literature: vanilla BO beats CMA-ES at small dimension + small budget — our SO(3) regime (3-D, tens of
queries). Caveat: BO degrades on multi-modal landscapes (ours is multi-modal). We test whether a minimal
GP-BO (sklearn GP + UCB) recovers SO(3) misorientation in FEWER label-free queries than random search /
CMA-ES. Compares at budgets {60,120,240} on ShapeNet part-seg SO(3). No new deps (sklearn only).

Run:  python -m src.eval.bo_query_efficiency --n 16
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg, make_input
from src.eval.metrics import mask_argmax, shape_miou
from src.zoo.canonicalize import Rso3, confidence_so3, zoo_canonicalize_so3_cma
from src.zoo.estimator import SPSA  # noqa: F401 (kept for parity)

B = np.pi


def miou_v(bb, xr, nr, clsv, cls, gt, v):
    R = Rso3(v)
    lg = bb(make_input(xr @ R.T, nr @ R.T, True), clsv).squeeze(0)
    return shape_miou(mask_argmax(lg, cls), gt, cls)


def bo_maximize(f, budget, n_init=12, kappa=2.0, seed=0):
    """Minimal GP-BO over v∈[-π,π]^3, maximize f. UCB acquisition over random candidates. Returns best x."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-B, B, size=(n_init, 3)); y = np.array([f(x) for x in X])
    for _ in range(max(budget - n_init, 0)):
        gp = GaussianProcessRegressor(ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5),
                                      normalize_y=True, alpha=1e-5, n_restarts_optimizer=0)
        gp.fit(X, y)
        cand = rng.uniform(-B, B, size=(512, 3))
        mu, sd = gp.predict(cand, return_std=True)
        xn = cand[(mu + kappa * sd).argmax()]
        X = np.vstack([X, xn]); y = np.append(y, f(xn))
    return X[y.argmax()]


def random_search(f, budget, seed):
    """Match the estimator-ablation random search: uniform axis on the sphere, angle ~ U(0, pi)."""
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=(budget, 3)); axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9
    X = axis * rng.uniform(0, np.pi, size=(budget, 1)); y = np.array([f(x) for x in X])
    return X[y.argmax()]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", type=int, nargs="+", default=[60, 120, 240])
    args = ap.parse_args()
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg(); rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), args.n, replace=False).tolist()

    res = {b: {"bo": [], "rand": [], "cma": []} for b in args.budgets}
    clean, corrupt = [], []
    for j, i in enumerate(idx):
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        clsv = s["cls"].view(1)
        clean.append(miou_v(bb, xyz, nrm, clsv, cls, gt, torch.zeros(3, device="cuda")))
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
        v0 = torch.tensor(axis * rng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)
        R0 = Rso3(v0); xr, nr = xyz @ R0.T, nrm @ R0.T
        corrupt.append(miou_v(bb, xr, nr, clsv, cls, gt, torch.zeros(3, device="cuda")))
        f = lambda x: confidence_so3(bb, xr, nr, clsv, torch.tensor(x, device="cuda", dtype=torch.float32)).item()  # noqa
        for b in args.budgets:
            vbo = bo_maximize(f, b, seed=100 + j)
            res[b]["bo"].append(miou_v(bb, xr, nr, clsv, cls, gt, torch.tensor(vbo, device="cuda", dtype=torch.float32)))
            vr = random_search(f, b, seed=200 + j)
            res[b]["rand"].append(miou_v(bb, xr, nr, clsv, cls, gt, torch.tensor(vr, device="cuda", dtype=torch.float32)))
            vc = zoo_canonicalize_so3_cma(bb, xr, nr, clsv, budget=b, restarts=2, seed=300 + j)[0]
            res[b]["cma"].append(miou_v(bb, xr, nr, clsv, cls, gt, vc))

    cl, co = np.mean(clean), np.mean(corrupt); gap = cl - co
    print(f"[BO query-efficiency @ ShapeNet SO(3), n={args.n}]  clean {cl:.3f} corrupt {co:.3f} gap {gap:.3f}")
    print(f"  {'budget':>7s} {'BO':>16s} {'random':>16s} {'CMA-ES':>16s}   (mIoU / %gap)")
    for b in args.budgets:
        def fmt(k):
            m = np.mean(res[b][k]); return f"{m:.3f} ({100*(m-co)/(gap+1e-9):.0f}%)"
        print(f"  {b:7d} {fmt('bo'):>16s} {fmt('rand'):>16s} {fmt('cma'):>16s}")


if __name__ == "__main__":
    main()
