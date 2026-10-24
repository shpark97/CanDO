"""Point sampling utilities (random, farthest point sampling). Each returns LongTensor indices (K,) into the input cloud."""
from __future__ import annotations

import torch


def random_sample(points: torch.Tensor, k: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Uniform random subset of k points (without replacement if k<=N, else with replacement)."""
    n = points.shape[0]
    if k >= n:
        return torch.arange(n, device=points.device)
    return torch.randperm(n, device=points.device, generator=generator)[:k]


def fps_sample(points: torch.Tensor, k: int) -> torch.Tensor:
    """Farthest point sampling via torch_cluster. Returns exactly min(k, N) indices."""
    from torch_cluster import fps

    n = points.shape[0]
    if k >= n:
        return torch.arange(n, device=points.device)
    ratio = float(k) / n
    idx = fps(points, ratio=ratio, random_start=False)  # ~ceil(ratio*N) indices
    return idx[:k]


SAMPLERS = {"random": random_sample, "fps": fps_sample}
