"""Export the interactive ShapeNetPart examples shown on the project page (docs/static/data/shapes.json).

For each listed category, test shapes are visited in a fixed random order. Each one is corrupted by the
headline random-axis SO(3) rotation (as in baseline_compare_partseg), segmented by the frozen PointNet++ as
is, and segmented again after CanDO canonicalization. The first shape whose mIoU improves by at least
--min_gain is exported: the misoriented input with the source prediction, the canonicalized input with the
CanDO prediction, and the upright shape with its ground-truth parts.

Run:  PYTHONPATH=. python scripts/make_project_page_data.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from src.data.shapenet_part import CATEGORIES, ShapeNetPart
from src.downstream.pointnet2_partseg import FrozenPartSeg, make_input
from src.eval.baseline_compare_partseg import ours_cma
from src.eval.metrics import PART_RANGES, mask_argmax, shape_miou
from src.zoo.canonicalize import Rso3

SHOW = ["Airplane", "Chair", "Table", "Lamp", "Motorbike", "Car", "Skateboard", "Pistol"]


def predict(bb, xyz, nrm, s, cls):
    lg = bb(make_input(xyz, nrm, True), s["cls"].view(1)).squeeze(0)
    pred = mask_argmax(lg, cls)
    return pred, shape_miou(pred, s["seg"].cuda(), cls)


def pack(xyz, labels, cls, sel):
    """Subsample, quantize to 1e-3, and store local part indices (0..P-1) for compact JSON."""
    p = (xyz[sel].cpu().numpy() * 1000).round().astype(int).reshape(-1).tolist()
    lo = PART_RANGES[cls][0]
    return {"xyz": p, "part": (labels[sel] - lo).cpu().numpy().astype(int).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/static/data/shapes.json")
    ap.add_argument("--points", type=int, default=2048)
    ap.add_argument("--min_gain", type=float, default=0.3)
    args = ap.parse_args()
    ds = ShapeNetPart("datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal", "test", npoints=None)
    bb = FrozenPartSeg()
    names = {c: n for _, n, c, _ in CATEGORIES}
    by_cat = {}
    for i, (_, c) in enumerate(ds.files):
        by_cat.setdefault(names[c], []).append(i)
    rng = np.random.default_rng(0)
    shapes = []
    for name in SHOW:
        for i in rng.permutation(by_cat[name]).tolist():
            s = ds[i]; cls = s["cls"].item()
            xyz, nrm = s["points"].cuda(), s["normals"].cuda()
            crng = np.random.default_rng(7_000_000 + i)                    # headline corruption draw
            axis = crng.normal(size=3); axis /= np.linalg.norm(axis) + 1e-9
            angle = crng.uniform(0.5, np.pi)
            R0 = Rso3(torch.tensor(axis * angle, device="cuda", dtype=torch.float32))
            xr, nr = xyz @ R0.T, nrm @ R0.T
            p_src, m_src = predict(bb, xr, nr, s, cls)
            v = ours_cma(bb, xr, nr, cls, seed=2_000_000 + i); R = Rso3(v)
            xc, nc = xr @ R.T, nr @ R.T
            p_ours, m_ours = predict(bb, xc, nc, s, cls)
            if m_ours - m_src < args.min_gain:
                continue
            sel = torch.from_numpy(np.random.default_rng(i).choice(len(xyz), min(args.points, len(xyz)),
                                                                   replace=False)).cuda()
            shapes.append({
                "category": name, "index": i, "parts": len(PART_RANGES[cls]),
                "angle_deg": round(float(np.degrees(angle)), 1),
                "miou_source": round(m_src, 3), "miou_cando": round(m_ours, 3),
                "input": pack(xr, p_src, cls, sel), "cando": pack(xc, p_ours, cls, sel),
                "target": pack(xyz, s["seg"].cuda(), cls, sel),
            })
            print(f"{name:>10s} #{i}: {np.degrees(angle):5.1f} deg  mIoU {m_src:.3f} -> {m_ours:.3f}", flush=True)
            break
    json.dump({"shapes": shapes}, open(args.out, "w"), separators=(",", ":"))
    print(f"wrote {len(shapes)} shapes -> {args.out}")


if __name__ == "__main__":
    main()
