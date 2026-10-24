"""Self-contained CanDO demo: no dataset, checkpoint, or GPU needed.

The "black box" is a toy scorer that we can only query: it returns a higher confidence the closer a
point cloud is to its canonical pose (an asymmetric three-armed object). We misorient the object with a
random SO(3) rotation and let CanDO recover the pose from confidence queries alone.

Run:  python examples/quickstart.py
"""
import math

import torch

from cando import cando, rodrigues, rotate


def make_object(n=1024, seed=0):
    """Three orthogonal arms of different lengths, so the canonical pose is unique."""
    g = torch.Generator().manual_seed(seed)
    arms = [torch.tensor([1.0, 0.15, 0.15]), torch.tensor([0.15, 0.6, 0.15]), torch.tensor([0.15, 0.15, 0.35])]
    pts = torch.cat([torch.rand(n // 3, 3, generator=g) * a for a in arms])
    return pts - pts.mean(0)


canonical = make_object()


def black_box_confidence(xs):
    """Score-only toy model: (M,N,3) rotated copies -> (M,) confidence in (0, 1]."""
    d = torch.cdist(xs, canonical.expand(xs.shape[0], -1, -1)).min(-1).values.mean(-1)
    return torch.exp(-5 * d)


def angle_between(Ra, Rb):
    c = ((Ra.T @ Rb).trace() - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, float(c)))))


torch.manual_seed(1)
axis = torch.nn.functional.normalize(torch.randn(3), dim=0)
R_corrupt = rodrigues(axis * 2.4)                       # about 137 degrees, a severe misorientation
observed = rotate(canonical, R_corrupt)


def objective(Rs):                                      # Rs: (M,3,3) candidate rotations
    return black_box_confidence(rotate(observed, Rs))   # one forward query per candidate


R, theta, info = cando(objective, budget=300, device="cpu")
print(f"confidence before: {float(objective(torch.eye(3)[None])):.3f}")
print(f"confidence after : {info['confidence']:.3f}  ({info['queries']} queries)")
print(f"residual pose error: {angle_between(R @ R_corrupt, torch.eye(3)):.1f} deg "
      f"(initial {angle_between(R_corrupt, torch.eye(3)):.1f} deg)")
