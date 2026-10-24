"""Generic shard aggregator: dedup per-shape rows by global idx, print mean (and %gap if clean/corrupt
present) for the requested keys.

Run:  PYTHONPATH=. python -m src.eval.aggregate_json "label" key1,key2,... shard_*.json
"""
from __future__ import annotations

import glob
import json
import sys

import numpy as np


def main():
    label = sys.argv[1]
    keys = sys.argv[2].split(",")
    paths = []
    for a in sys.argv[3:]:
        paths.extend(glob.glob(a))
    seen = {}
    for p in paths:
        d = json.load(open(p))
        for k, i in enumerate(d["idx"]):
            if i not in seen:
                seen[i] = {key: d[key][k] for key in keys}
    rows = list(seen.values())
    n = len(rows)
    m = {key: float(np.mean([r[key] for r in rows])) for key in keys}
    sem = {key: float(np.std([r[key] for r in rows]) / (n ** 0.5)) for key in keys}
    has_gap = "clean" in m and "corrupt" in m
    gap = (m["clean"] - m["corrupt"] + 1e-9) if has_gap else None
    print(f"[{label} FULL n={n}]")
    for key in keys:
        tag = f"   ({100*(m[key]-m['corrupt'])/gap:+.0f}% gap)" if (has_gap and key not in ("clean",)) else ""
        print(f"  {key:9s} = {m[key]:.3f} +/- {sem[key]:.3f} (SEM){tag}")


if __name__ == "__main__":
    main()
