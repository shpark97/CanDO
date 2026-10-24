"""Export the "watch the search" and "confidence landscape" demos of the project page
(docs/static/data/explore.json).

For each listed example of docs/static/data/shapes.json (same shape, same headline SO(3) corruption):
  landscape  the corrupted input is rotated back about the corruption axis by phi in [0, 360); at every
             step we record the label-free confidence J, the true mIoU (display only), and the predicted
             parts. phi equal to the corruption angle is exactly the upright pose.
  search     one CanDO run (CMA-ES, population 12, sigma0 1.0, 300 queries, 2 restarts, as in Table 2) with
             every candidate's axis-angle vector and confidence, plus the best-so-far pose, its predicted
             parts, and its mIoU after each generation.
Part labels are stored as digit strings to keep the file small.

Run:  PYTHONPATH=. python scripts/make_project_page_explore.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg
from src.eval.baseline_compare_partseg import batched_conf
from src.eval.metrics import PART_RANGES, mask_argmax, shape_miou
from src.zoo.canonicalize import Rso3


def corrupt(xyz, nrm, i):
    crng = np.random.default_rng(7_000_000 + i)                       # headline corruption draw
    axis = crng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
    angle = crng.uniform(0.5, np.pi)
    R0 = Rso3(torch.tensor(axis * angle, device="cuda", dtype=torch.float32))
    return xyz @ R0.T, nrm @ R0.T, axis, angle


def score(bb, xr, nr, cls, gt, vs, sel):
    """Confidence, mIoU, and predicted-part strings for candidate axis-angle vectors vs (batched)."""
    conf, miou, parts = [], [], []
    lo = PART_RANGES[cls][0]
    for c in range(0, len(vs), 40):
        lg, cf = batched_conf(bb, xr, nr, cls, vs[c:c + 40])
        for m in range(lg.shape[0]):
            pred = mask_argmax(lg[m], cls)
            miou.append(round(shape_miou(pred, gt, cls), 3))
            parts.append("".join(str(int(p)) for p in (pred[sel] - lo).tolist()))
        conf += [round(float(x), 4) for x in cf]
    return conf, miou, parts


def search(bb, xr, nr, cls, gt, sel, seed, budget=300, popsize=12, restarts=2, sigma0=1.0):
    """CanDO's Table 2 search (ours_cma), logged generation by generation."""
    import cma
    cands, gens = [], []
    best = (-1.0, np.zeros(3))
    for r in range(restarts):
        x0 = (np.random.default_rng(seed + r).normal(size=3) * 0.3).tolist()
        es = cma.CMAEvolutionStrategy(x0, sigma0, {"maxfevals": budget // restarts, "popsize": popsize,
                                                   "verbose": -9, "seed": seed + r + 1})
        while not es.stop():
            sols = es.ask()
            vs = [torch.tensor(s, device="cuda", dtype=torch.float32) for s in sols]
            _, conf = batched_conf(bb, xr, nr, cls, vs)
            conf = conf.tolist()
            es.tell(sols, [-c for c in conf])
            for s, c in zip(sols, conf):
                cands.append([round(float(v), 3) for v in s] + [round(c, 4), len(gens)])
                if c > best[0]:
                    best = (c, np.asarray(s))
            gens.append({"restart": r, "best_theta": [round(float(v), 4) for v in best[1]],
                         "best_conf": round(best[0], 4), "queries": len(cands)})
    bv = [torch.tensor(g["best_theta"], device="cuda", dtype=torch.float32) for g in gens]
    _, miou, parts = score(bb, xr, nr, cls, gt, bv, sel)
    for g, m, p in zip(gens, miou, parts):
        g["miou"], g["parts"] = m, p
    return cands, gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes_json", default="docs/static/data/shapes.json")
    ap.add_argument("--out", default="docs/static/data/explore.json")
    ap.add_argument("--show", nargs="+", default=["Airplane", "Chair", "Table"])
    ap.add_argument("--points", type=int, default=1536)
    ap.add_argument("--steps", type=int, default=120)
    args = ap.parse_args()
    picked = {s["category"]: s["index"] for s in json.load(open(args.shapes_json))["shapes"]}
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg()
    out = []
    for name in args.show:
        i = picked[name]; s = ds[i]; cls = s["cls"].item()
        xyz, nrm, gt = s["points"].cuda(), s["normals"].cuda(), s["seg"].cuda()
        xr, nr, axis, angle = corrupt(xyz, nrm, i)
        sel = torch.from_numpy(np.random.default_rng(i).choice(len(xyz), min(args.points, len(xyz)),
                                                               replace=False)).cuda()
        phis = np.linspace(0, 2 * np.pi, args.steps, endpoint=False)
        vs = [torch.tensor(-phi * axis, device="cuda", dtype=torch.float32) for phi in phis]
        conf, miou, parts = score(bb, xr, nr, cls, gt, vs, sel)
        cands, gens = search(bb, xr, nr, cls, gt, sel, seed=2_000_000 + i)
        out.append({
            "category": name, "index": i, "parts": len(PART_RANGES[cls]),
            "xyz": (xr[sel].cpu().numpy() * 1000).round().astype(int).reshape(-1).tolist(),
            "axis": [round(float(a), 5) for a in axis], "angle_deg": round(float(np.degrees(angle)), 1),
            "landscape": {"phi_deg": [round(float(np.degrees(p)), 1) for p in phis], "conf": conf, "miou": miou,
                          "parts": parts},
            "search": {"cands": cands, "gens": gens},
        })
        k = int(np.argmax(conf))
        print(f"{name:>9s} #{i}: angle {np.degrees(angle):.0f} deg, landscape peak at {np.degrees(phis[k]):.0f} deg; "
              f"search {len(cands)} queries, {len(gens)} generations, final mIoU {gens[-1]['miou']:.3f}", flush=True)
    json.dump({"examples": out}, open(args.out, "w"), separators=(",", ":"))
    print(f"wrote {len(out)} examples -> {args.out}")


if __name__ == "__main__":
    main()
