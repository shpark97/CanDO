"""Zeroth-order gradient estimators behind one interface (the local baselines of Table 5).

    estimate(theta, eval_fn) -> grad_estimate (same shape as theta)

`eval_fn(theta) -> scalar loss` uses only black-box forward passes (no internal gradients). Each
estimator perturbs θ and combines loss values. Query cost is whatever eval_fn incurs (tracked by
the black-box wrapper); `last_n_evals` records how many eval_fn calls this estimate used.

Implemented: SPSA (2 evals/step, dimension-independent), NES (2m evals, antithetic).
"""
from __future__ import annotations

from typing import Callable

import torch

EvalFn = Callable[[torch.Tensor], torch.Tensor]


class Estimator:
    last_n_evals: int = 0

    def estimate(self, theta: torch.Tensor, eval_fn: EvalFn) -> torch.Tensor:  # pragma: no cover
        raise NotImplementedError


class SPSA(Estimator):
    """Simultaneous Perturbation Stochastic Approximation. ĝ = (L+ - L-)/(2c) * Δ, Δ∈{±1}^P."""

    def __init__(self, c: float = 0.05, avg: int = 1, seed: int | None = None):
        self.c = c
        self.avg = avg  # average over `avg` independent ± pairs (variance reduction)
        self.gen = None
        if seed is not None:
            self.gen = torch.Generator()  # device set lazily in estimate

    def estimate(self, theta: torch.Tensor, eval_fn: EvalFn) -> torch.Tensor:
        grad = torch.zeros_like(theta)
        for _ in range(self.avg):
            delta = torch.randint(0, 2, theta.shape, device=theta.device).float() * 2 - 1  # ±1
            lp = eval_fn(theta + self.c * delta)
            lm = eval_fn(theta - self.c * delta)
            grad += (lp - lm) / (2 * self.c) * delta  # 1/delta == delta for ±1
        self.last_n_evals = 2 * self.avg
        return grad / self.avg


class NES(Estimator):
    """Gaussian-smoothing natural ES with antithetic sampling.
    ĝ = (1/(σ m)) Σ_j ½(L(θ+σu_j) - L(θ-σu_j)) u_j ,  u_j ~ N(0, I)."""

    def __init__(self, sigma: float = 0.05, m: int = 8, fitness_shaping: bool = False):
        self.sigma = sigma
        self.m = m
        self.fitness_shaping = fitness_shaping

    def estimate(self, theta: torch.Tensor, eval_fn: EvalFn) -> torch.Tensor:
        us, diffs = [], []
        for _ in range(self.m):
            u = torch.randn_like(theta)
            lp = eval_fn(theta + self.sigma * u)
            lm = eval_fn(theta - self.sigma * u)
            us.append(u)
            diffs.append(0.5 * (lp - lm))
        self.last_n_evals = 2 * self.m
        U = torch.stack(us)            # (m, P)
        d = torch.stack(diffs)         # (m,)
        if self.fitness_shaping:       # rank-normalize the half-differences
            ranks = d.argsort().argsort().float()
            d = (ranks / (self.m - 1) - 0.5) if self.m > 1 else d
        return (d.unsqueeze(1) * U).mean(0) / self.sigma


class Coordinate(Estimator):
    """Coordinate / central finite-difference. ĝ_k = (L(θ+ε e_k) - L(θ-ε e_k)) / (2ε) for each dim.
    Exact-ish when θ is very low-dim; 2P evals/step (no dimension-independence, unlike SPSA)."""

    def __init__(self, eps: float = 0.05):
        self.eps = eps

    def estimate(self, theta: torch.Tensor, eval_fn: EvalFn) -> torch.Tensor:
        grad = torch.zeros_like(theta)
        flat = theta.reshape(-1)
        for k in range(flat.numel()):
            e = torch.zeros_like(flat); e[k] = self.eps
            lp = eval_fn((flat + e).view_as(theta))
            lm = eval_fn((flat - e).view_as(theta))
            grad.reshape(-1)[k] = (lp - lm) / (2 * self.eps)
        self.last_n_evals = 2 * flat.numel()
        return grad


class REINFORCE(Estimator):
    """Score-function / policy-gradient estimator over a Gaussian search policy π_μ = N(μ, σ²I).
    For J(μ)=E_{v~π}[reward], ∇_μ J = E[(R-b)(v-μ)/σ²]; with eval_fn returning a LOSS we descend on
    ∇_μ E[L] = (1/m)Σ (L_j - b) ε_j / σ, b = batch-mean baseline (variance reduction). m evals/step
    (non-antithetic), the classic REINFORCE. Distinct from NES (antithetic, natural-gradient form)."""

    def __init__(self, sigma: float = 0.15, m: int = 8):
        self.sigma = sigma
        self.m = m

    def estimate(self, theta: torch.Tensor, eval_fn: EvalFn) -> torch.Tensor:
        eps, losses = [], []
        for _ in range(self.m):
            e = torch.randn_like(theta)
            losses.append(eval_fn(theta + self.sigma * e))
            eps.append(e)
        self.last_n_evals = self.m
        L = torch.stack([l.reshape(()) for l in losses])  # (m,)
        b = L.mean()                                       # baseline
        E = torch.stack(eps)                               # (m, P)
        return ((L - b).reshape(-1, *([1] * (E.dim() - 1))) * E).mean(0) / self.sigma


ESTIMATORS = {"spsa": SPSA, "nes": NES, "coordinate": Coordinate, "reinforce": REINFORCE}
