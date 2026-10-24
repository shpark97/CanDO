"""DFO-based label-free test-time canonicalization for a frozen 3D model.

Inputs are (deliberately) mis-oriented. We recover the orientation by gradient-free optimization
of a rotation theta against a BLACK-BOX, LABEL-FREE objective: maximize the frozen model's mean
max-probability (confidence). No GT, no model gradients.

Run:  python -m src.zoo.canonicalize --n_shapes 100 --restarts 3 --steps 25
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg, make_input
from src.eval.metrics import mask_argmax, shape_miou
from src.zoo.estimator import SPSA


def Rz(angle: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(angle), torch.sin(angle)
    z, o = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack([torch.stack([c, -s, z]), torch.stack([s, c, z]), torch.stack([z, z, o])]).squeeze(-1)


def Rso3(v: torch.Tensor) -> torch.Tensor:
    """Axis-angle 3-vector -> rotation matrix (Rodrigues). |v|=angle, v/|v|=axis. Full SO(3)."""
    theta = v.norm() + 1e-12
    k = v / theta
    K = torch.stack([
        torch.stack([torch.zeros((), device=v.device), -k[2], k[1]]),
        torch.stack([k[2], torch.zeros((), device=v.device), -k[0]]),
        torch.stack([-k[1], k[0], torch.zeros((), device=v.device)]),
    ])
    I = torch.eye(3, device=v.device)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def confidence_so3(bb, xyz, nrm, cls_view, v: torch.Tensor) -> torch.Tensor:
    R = Rso3(v)
    lg = bb(make_input(xyz @ R.T, nrm @ R.T, True), cls_view).squeeze(0)
    return lg.exp().max(-1).values.mean()


def zoo_canonicalize_so3(bb, xyz, nrm, cls_view, estimator, restarts=4, steps=20, lr=0.2, seed=0):
    """Recover an SO(3) rotation by gradient-free max-confidence. `estimator` is a fresh zeroth-order
    estimator (SPSA/NES). SO(3) is 3-D & multi-modal so a grid is infeasible -> zeroth-order search + random
    restarts is the natural solver. Returns (best_v, n_queries)."""
    neg = lambda v: -confidence_so3(bb, xyz, nrm, cls_view, v)  # noqa: E731
    rng = np.random.default_rng(seed)
    best_v, best_c, nq = torch.zeros(3, device=xyz.device), -1.0, 0
    for _ in range(restarts):
        axis = rng.normal(size=3); axis /= (np.linalg.norm(axis) + 1e-9)
        v = torch.tensor(axis * rng.uniform(0, np.pi), device=xyz.device, dtype=torch.float32)
        for t in range(steps):
            g = estimator.estimate(v, neg)
            v = v - (lr / (1 + 0.1 * t)) * g
            nq += estimator.last_n_evals
        c = confidence_so3(bb, xyz, nrm, cls_view, v).item(); nq += 1
        if c > best_c:
            best_c, best_v = c, v.detach().clone()
    return best_v, nq


def zoo_canonicalize_so3_cma(bb, xyz, nrm, cls_view, budget=300, sigma0=1.2, restarts=2, seed=0):
    """CMA-ES (population gradient-free ES) over SO(3) axis-angle v∈R^3, label-free max-confidence.
    A proper global ES — the right tool for the 3-D multi-modal landscape where SPSA/random struggle."""
    import cma

    best_v, best_c, nq = torch.zeros(3, device=xyz.device), -1.0, 0
    for r in range(restarts):
        x0 = (np.random.default_rng(seed + r).normal(size=3) * 0.3).tolist()
        es = cma.CMAEvolutionStrategy(x0, sigma0, {
            "maxfevals": budget // restarts, "popsize": 10, "verbose": -9, "seed": seed + r + 1})
        while not es.stop():
            sols = es.ask()
            fits = []
            for x in sols:
                v = torch.tensor(x, device=xyz.device, dtype=torch.float32)
                fits.append(-confidence_so3(bb, xyz, nrm, cls_view, v).item())
                nq += 1
            es.tell(sols, fits)
        v = torch.tensor(es.result.xbest, device=xyz.device, dtype=torch.float32)
        c = confidence_so3(bb, xyz, nrm, cls_view, v).item(); nq += 1
        if c > best_c:
            best_c, best_v = c, v
    return best_v, nq


def confidence(bb, xyz, nrm, cls_view, angle: torch.Tensor) -> torch.Tensor:
    """Label-free objective: mean max-softmax prob after rotating by `angle` (z-axis)."""
    R = Rz(angle)
    lg = bb(make_input(xyz @ R.T, nrm @ R.T, True), cls_view).squeeze(0)  # (N,50) log-prob
    return lg.exp().max(-1).values.mean()


def zoo_canonicalize(bb, xyz, nrm, cls_view, coarse=16, steps=15, c=0.08, lr=0.15, seed=0):
    """Coarse-to-fine gradient-free: a coarse global scan (confidence is 1-D, periodic, multi-modal)
    finds the basin, then SPSA refines to the continuous optimum within it. Label-free throughout."""
    neg_conf = lambda th: -confidence(bb, xyz, nrm, cls_view, th)  # noqa: E731
    # coarse global scan
    angs = torch.linspace(0, 2 * np.pi, coarse + 1, device=xyz.device)[:-1]
    confs = torch.tensor([confidence(bb, xyz, nrm, cls_view, a.view(1)).item() for a in angs])
    theta = angs[int(confs.argmax())].view(1).clone()
    n_eval = coarse
    # SPSA local refinement (continuous angle)
    est = SPSA(c=c, avg=1)
    for t in range(steps):
        grad = est.estimate(theta, neg_conf)
        theta = theta - (lr / (1 + 0.1 * t)) * grad
        n_eval += est.last_n_evals
    return theta, n_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    ap.add_argument("--n_shapes", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    ds = ShapeNetPart(args.root, split="test", npoints=None)
    bb = FrozenPartSeg()
    rng = np.random.default_rng(args.seed)
    idx = list(range(len(ds))) if args.n_shapes < 0 else rng.choice(len(ds), args.n_shapes, replace=False).tolist()

    misor, zoo, oracle, n_evals = [], [], [], []
    grid = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    for i in idx:
        s = ds[i]; xyz, nrm, gt, cls = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda(), s["cls"].item()
        a0 = rng.uniform(0, 2 * np.pi)
        R0 = Rz(torch.tensor([a0], device="cuda", dtype=torch.float32))
        xr, nr = xyz @ R0.T, nrm @ R0.T                         # mis-oriented input
        clsv = s["cls"].view(1)

        def miou_at(angle):
            R = Rz(torch.tensor([angle], device="cuda", dtype=torch.float32))
            lg = bb(make_input(xr @ R.T, nr @ R.T, True), clsv).squeeze(0)
            return shape_miou(mask_argmax(lg, cls), gt, cls)

        misor.append(miou_at(0.0))
        ang, ne = zoo_canonicalize(bb, xr, nr, clsv, coarse=16, steps=args.steps, seed=i)
        zoo.append(miou_at(ang.item())); n_evals.append(ne)
        oracle.append(max(miou_at(a) for a in grid))

    print(f"[zeroth-order canonicalization @ ShapeNet, n={len(idx)}, mis-oriented inputs]")
    print(f"  no-recovery        mIoU = {np.mean(misor):.4f}")
    print(f"  SPSA (label-free) = {np.mean(zoo):.4f}   (avg {np.mean(n_evals):.0f} black-box queries/shape)")
    print(f"  oracle (grid-16, GT)   = {np.mean(oracle):.4f}")
    print(f"  -> SPSA recovers {np.mean(zoo)-np.mean(misor):+.4f} mIoU label-free "
          f"({100*(np.mean(zoo)-np.mean(misor))/(np.mean(oracle)-np.mean(misor)):.0f}% of oracle gap)")


if __name__ == "__main__":
    main()
