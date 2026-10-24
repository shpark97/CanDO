"""CanDO: black-box pose canonicalization by derivative-free optimization (Algorithm 1 of the paper).

Model-agnostic entry point. The caller supplies `objective`, which maps a batch of rotation matrices
(M, 3, 3) to the frozen model's label-free confidence (M,) for the input rotated by each matrix, using
forward queries only. CanDO searches axis-angle vectors theta in R^3 with CMA-ES, keeps every scored
candidate (identity included), and returns the one maximizing  J(theta) - lam * ||theta||.

Example (part segmentation, points P (N,3) and normals Nr (N,3) as row vectors):

    def objective(Rs):                                   # Rs: (M,3,3)
        xs = torch.einsum("mij,nj->mni", Rs, P)          # rotated copies of the input
        ns = torch.einsum("mij,nj->mni", Rs, Nr)
        probs = frozen_model(xs, ns)                     # (M,N,C) class probabilities
        return probs.max(-1).values.mean(-1)             # mean max-probability per candidate

    R, theta, info = cando(objective)
    pred = frozen_model(P @ R.T, Nr @ R.T)              # prediction in the canonicalized frame

The per-experiment scripts in src/eval/ inline task-specific variants of this loop (batched scoring,
masked confidence, per-task budgets); see README.md for the commands behind each table.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def rodrigues(theta: torch.Tensor) -> torch.Tensor:
    """Axis-angle vectors (..., 3) -> rotation matrices (..., 3, 3), with the standard limit at 0."""
    phi = theta.norm(dim=-1, keepdim=True).unsqueeze(-1)                     # (...,1,1)
    x, y, z = theta.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*theta.shape[:-1], 3, 3)
    eye = torch.eye(3, device=theta.device, dtype=theta.dtype).expand_as(K)
    small = phi < 1e-8
    phi_safe = torch.where(small, torch.ones_like(phi), phi)
    a = torch.where(small, torch.ones_like(phi), torch.sin(phi_safe) / phi_safe)
    b = torch.where(small, 0.5 * torch.ones_like(phi), (1 - torch.cos(phi_safe)) / phi_safe ** 2)
    return eye + a * K + b * (K @ K)


def rotate(points: torch.Tensor, Rs: torch.Tensor) -> torch.Tensor:
    """Rotate row-vector points (N,3) by one (3,3) or a batch (M,3,3) of rotations -> (N,3) or (M,N,3)."""
    if Rs.dim() == 2:
        return points @ Rs.T
    return torch.einsum("mij,nj->mni", Rs, points)


def max_prob_confidence(probs: torch.Tensor) -> torch.Tensor:
    """Confidence surrogate J of Eq. (4) from class probabilities.

    (M, C) for classification -> maximum probability per candidate.
    (M, N, C) for segmentation -> mean over points of the per-point maximum probability.
    """
    conf = probs.max(-1).values
    return conf.mean(-1) if conf.dim() > 1 else conf


@torch.no_grad()
def cando(objective: Callable[[torch.Tensor], torch.Tensor], budget: int = 300, popsize: int = 10,
          sigma0: float = 1.2, restarts: int = 2, lam: float = 0.0, seed: int = 0,
          device: str | torch.device | None = None):
    """Search SO(3) for the rotation that maximizes `objective` using at most `budget` forward queries.

    Args:
        objective: maps rotation matrices (M,3,3) to confidences (M,), one forward query per matrix.
        budget: total number of scored rotations, the identity included.
        popsize: CMA-ES population, i.e. the number of rotations scored per batched call.
        sigma0: initial CMA-ES step size in axis-angle units (radians).
        restarts: independent CMA-ES runs sharing the budget.
        lam: weight of the identity prior of Eq. (5); 0 selects the most confident rotation.
        seed: seed of the CMA-ES runs, for reproducible searches.
        device: device of the rotation matrices passed to `objective` (default: cuda if available).

    Returns (R, theta, info): the selected rotation matrix (3,3), its axis-angle vector (3,), and a dict
    with the selected confidence and the number of queries used.
    """
    import cma

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    zero = torch.zeros(1, 3, device=device)
    thetas, scores = [zero[0]], [float(objective(rodrigues(zero))[0])]          # identity first
    per_restart = max(popsize, (budget - 1) // restarts)
    for r in range(restarts):
        x0 = (np.random.default_rng(seed + r).normal(size=3) * 0.3).tolist()
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"maxfevals": per_restart, "popsize": popsize,
                                                   "verbose": -9, "seed": seed + r + 1})
        while not es.stop() and len(scores) < budget:
            sols = es.ask()[: budget - len(scores)]
            th = torch.tensor(np.asarray(sols), device=device, dtype=torch.float32)
            conf = torch.as_tensor(objective(rodrigues(th))).float().cpu().tolist()
            if len(sols) == popsize:
                es.tell(sols, [-c for c in conf])
            thetas.extend(th); scores.extend(conf)
    th = torch.stack(thetas)
    sel = torch.tensor(scores, device=device) - lam * th.norm(dim=-1)
    k = int(sel.argmax())
    return rodrigues(th[k]), th[k], {"confidence": scores[k], "queries": len(scores)}
