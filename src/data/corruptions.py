"""Official PointCloud-C atomic corruptions (Ren et al., ICML 2022), wrapped for this code base.

7 atomic corruptions x 5 severities, imported directly from third_party/PointCloud-C so our numbers
follow the standard benchmark exactly. Taxonomy:
  - transformation (pose-ish): rotate (SO(3) up to ±30°), scale (anisotropic)
  - noise:                     jitter, add_global, add_local
  - density:                   dropout_global, dropout_local
CanDO targets the pose corruption (rotate); density and noise corruptions are outside what a rotation can
fix (Tables 7 and 8).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_PCC = os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "PointCloud-C", "build")
if os.path.abspath(_PCC) not in sys.path:
    sys.path.insert(0, os.path.abspath(_PCC))

from corrupt_utils import (  # noqa: E402
    corrupt_add_global, corrupt_add_local, corrupt_dropout_global, corrupt_dropout_local,
    corrupt_jitter, corrupt_rotate, corrupt_scale,
)

CORRUPTIONS = {
    "scale": corrupt_scale,
    "jitter": corrupt_jitter,
    "rotate": corrupt_rotate,
    "dropout_global": corrupt_dropout_global,
    "dropout_local": corrupt_dropout_local,
    "add_global": corrupt_add_global,
    "add_local": corrupt_add_local,
}
FAMILY = {"scale": "transform", "rotate": "transform", "jitter": "noise",
          "add_global": "noise", "add_local": "noise",
          "dropout_global": "density", "dropout_local": "density"}
POSE_CORRUPTIONS = ["rotate", "scale"]  # the ones a geometric canonicalization could repair


def corrupt(pcd: np.ndarray, ctype: str, level: int) -> np.ndarray:
    """pcd: (N,3) float32 unit-normalized. level in 0..4. Returns corrupted (M,3)."""
    return CORRUPTIONS[ctype](pcd.astype("float32"), level).astype("float32")


def fix_npoints(pcd: np.ndarray, n: int = 1024) -> np.ndarray:
    """Pad (repeat) or subsample to exactly n points so a fixed-N frozen model can ingest it.
    (Density corruptions change the count; the model's internal FPS needs N >= its first npoint.)"""
    m = pcd.shape[0]
    if m == n:
        return pcd
    if m > n:
        idx = np.random.default_rng(0).choice(m, n, replace=False)
        return pcd[idx]
    pad = np.random.default_rng(0).choice(m, n - m, replace=True)
    return np.concatenate([pcd, pcd[pad]], axis=0)
