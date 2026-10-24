"""Aggregate the 3-seed Table 1 runs into table cells: mean +/- std over the three corruption seeds for
corrupted / recovered (ours) / oracle, plus % of the oracle gap recovered.

Reads the per-shard dumps written by the commands in README.md (PointNet++: runs/s3_<task>_c<cs>_g<g>.json,
PointNeXt-S: runs/pnx_ps/<transform>_c<cs>_g<g>.json and runs/mn_pnx_c<cs>.json). Rows whose dumps are
missing are skipped.

Run:  PYTHONPATH=. python -m src.eval.agg_seed3
"""
from __future__ import annotations

import glob
import json

import numpy as np

TASKS = {  # row -> glob pattern with {cs} = corruption seed
    "PointNet++ ShapeNetPart azimuth": "runs/s3_azimuth_c{cs}_g*.json",
    "PointNet++ ShapeNetPart SO(3)": "runs/s3_partseg_c{cs}_g*.json",
    "PointNet++ ModelNet40 SO(3)": "runs/s3_cls_c{cs}_g*.json",
    "PointNeXt-S ShapeNetPart azimuth": "runs/pnx_ps/azimuth_c{cs}_g*.json",
    "PointNeXt-S ShapeNetPart SO(3)": "runs/pnx_ps/so3_c{cs}_g*.json",
    "PointNeXt-S ModelNet40 SO(3)": "runs/mn_pnx_c{cs}.json",
}
KEYS = ["clean", "corrupt", "ours", "oracle"]


def main():
    for task, pattern in TASKS.items():
        seed_means = {k: [] for k in KEYS}
        nseed, n = 0, 0
        for cs in range(3):
            files = sorted(glob.glob(pattern.format(cs=cs)))
            if not files:
                continue
            rows = {}                                          # dedup shards by global shape index
            for fp in files:
                o = json.load(open(fp))
                for j, i in enumerate(o["idx"]):
                    rows[i] = {k: o[k][j] for k in KEYS}
            for k in KEYS:
                seed_means[k].append(float(np.mean([r[k] for r in rows.values()])))
            nseed += 1; n = len(rows)
        if nseed == 0:
            print(f"[{task}] no dumps found"); continue
        print(f"[{task}]  {nseed} seeds, n={n}")
        cells = {}
        for k in KEYS:
            mu, sd = np.mean(seed_means[k]), np.std(seed_means[k])
            cells[k] = (mu, sd)
            print(f"    {k:8s} = {mu:.3f} +/- {sd:.3f}   (seeds: {[round(x,3) for x in seed_means[k]]})")
        c, r, o = cells["corrupt"], cells["ours"], cells["oracle"]
        gap = r[0] - c[0]; full = o[0] - c[0] + 1e-9
        print(f"    -> Table 1 row:  ${c[0]:.3f}{{\\pm}}{c[1]:.3f}$ & "
              f"$\\mathbf{{{r[0]:.3f}}}{{\\pm}}{r[1]:.3f}$ & ${o[0]:.3f}{{\\pm}}{o[1]:.3f}$ & "
              f"{100*gap/full:.0f}\\%")


if __name__ == "__main__":
    main()
