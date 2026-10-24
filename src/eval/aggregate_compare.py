"""Aggregate sharded baseline_compare_partseg runs into the final full-test table.

Each shard wrote a JSON via --dump with per-shape lists; we concatenate (de-duplicating by global idx in
case of overlap) and recompute the means and percent-of-gap, identical to a single-process full run.

Run:  PYTHONPATH=. python -m src.eval.aggregate_compare runs/cmpshard_*.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

KEYS = ["clean", "corrupt", "pca", "pcaconf", "tent", "memo", "ours", "oracle"]


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: aggregate_compare <shard1.json> <shard2.json> ..."); return
    seen = {}
    for p in paths:
        d = json.load(open(p))
        for k, i in enumerate(d["idx"]):
            if i in seen:
                continue
            seen[i] = {key: d[key][k] for key in KEYS}
    rows = list(seen.values())
    n = len(rows)
    m = {key: float(np.mean([r[key] for r in rows])) for key in KEYS}
    sem = {key: float(np.std([r[key] for r in rows]) / (n ** 0.5)) for key in KEYS}
    gap = m["clean"] - m["corrupt"]
    print(f"[Competitive @ ShapeNet part-seg SO(3), FULL test n={n}]  "
          f"clean {m['clean']:.3f} corrupt {m['corrupt']:.3f} gap {gap:.3f}")
    for key in ["corrupt", "pca", "pcaconf", "tent", "memo", "ours", "oracle"]:
        pct = 100 * (m[key] - m["corrupt"]) / (gap + 1e-9)
        print(f"  {key:8s} = {m[key]:.3f} +/- {sem[key]:.3f} (SEM)   ({pct:+.0f}% gap)")


if __name__ == "__main__":
    main()
