"""Aggregate sharded baseline_compare_s3dis runs into the final full-test table.

Each shard wrote a JSON via --dump with per-block lists; we concatenate (de-duplicating by global candidate
idx) and recompute means + SEM, identical to a single-process full run.

Run:  PYTHONPATH=. python -m src.eval.aggregate_compare_s3dis runs/s3dis_cmpshard_*.json
"""
from __future__ import annotations

import json
import sys

import numpy as np

KEYS = ["clean", "corrupt", "pca", "pcaconf", "dua", "tent", "memo", "grad", "ours", "oracle"]


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: aggregate_compare_s3dis <shard1.json> <shard2.json> ..."); return
    seen = {}
    for p in paths:
        d = json.load(open(p))
        for k, i in enumerate(d["idx"]):
            if i in seen:
                continue
            seen[i] = {key: d[key][k] for key in KEYS if key in d and k < len(d[key])}
    rows = [r for r in seen.values() if "clean" in r]
    n = len(rows)
    present = [key for key in KEYS if all(key in r for r in rows)]
    m = {key: float(np.mean([r[key] for r in rows])) for key in present}
    sem = {key: float(np.std([r[key] for r in rows]) / (n ** 0.5)) for key in present}
    gap = m["clean"] - m["corrupt"]
    print(f"[Competitive @ S3DIS sem-seg TILT, FULL test n={n} blocks]  "
          f"clean {m['clean']:.3f} corrupt {m['corrupt']:.3f} gap {gap:.3f}")
    for key in present[1:]:
        pct = 100 * (m[key] - m["corrupt"]) / (gap + 1e-9)
        print(f"  {key:8s} = {m[key]:.3f} +/- {sem[key]:.3f} (SEM)   ({pct:+.0f}% gap)")


if __name__ == "__main__":
    main()
