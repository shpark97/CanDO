"""Equivariant part-segmentation no-op check.

A frozen Vector-Neurons (VN) part-segmentation model is rotation-invariant by construction, so an SO(3)
corruption leaves its predictions unchanged and CanDO reduces to the identity: clean = corrupt = recovered.
Table 9: the contrast is sharp because the non-equivariant PointNet++ part-seg model collapses under SO(3) and CanDO recovers a
large share of the gap.

Run: PYTHONPATH=. python -m src.eval.vn_partseg_noop --model vn_pointnet_partseg --ckpt <path> --n 256
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from src.data.shapenet_part import ShapeNetPart
from src.eval.metrics import mask_argmax, shape_miou

_VNN = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "vnn"))
for _p in (_VNN, os.path.join(_VNN, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

NUM_CAT = 16
LAM = 0.05  # identity-prior weight


class _Args:  # vnn model construction args
    n_knn = 40
    pooling = "mean"


def Rso3(v: torch.Tensor) -> torch.Tensor:
    th = v.norm() + 1e-12
    k = v / th
    K = torch.zeros(3, 3, device=v.device, dtype=v.dtype)
    K[0, 1] = -k[2]; K[0, 2] = k[1]; K[1, 0] = k[2]; K[1, 2] = -k[0]; K[2, 0] = -k[1]; K[2, 1] = k[0]
    I = torch.eye(3, device=v.device, dtype=v.dtype)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="vn_pointnet_partseg | vn_dgcnn_partseg")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--M", type=int, default=128, help="candidate pool size (incl. identity)")
    ap.add_argument("--npoints", type=int, default=2048)
    ap.add_argument("--root", default="datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    args = ap.parse_args()

    net = __import__(args.model).get_model(_Args(), num_part=50, normal_channel=False).cuda()
    net.load_state_dict(torch.load(args.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    net.eval()

    @torch.no_grad()
    def logits(xyz, cls_id):                       # xyz (N,3) -> (N,50) log-prob
        pts = (xyz - xyz.mean(0, keepdim=True)).T.unsqueeze(0).contiguous()   # (1,3,N)
        oh = torch.eye(NUM_CAT, device=xyz.device)[cls_id].view(1, 1, NUM_CAT)
        return net(pts, oh)[0].squeeze(0)          # (N,50)

    def conf(xyz, cls_id):                          # mean max-softmax (label-free objective)
        return logits(xyz, cls_id).exp().max(-1).values.mean().item()

    ds = ShapeNetPart(args.root, "test", npoints=args.npoints)
    n = len(ds) if args.n <= 0 else min(args.n, len(ds))   # n<=0 -> full test set
    idx = np.random.default_rng(0).choice(len(ds), n, replace=False).tolist()
    res = {k: [] for k in ("clean", "corrupt", "recovered")}
    for j, i in enumerate(idx):
        s = ds[i]; xyz = s["points"].float().cuda(); gt = s["seg"].cuda(); cls = int(s["cls"])
        res["clean"].append(shape_miou(mask_argmax(logits(xyz, cls), cls), gt, cls))
        # SO(3) corruption (axis-uniform, angle U(0.5,pi)) matching the deployed-headline convention
        crng = np.random.default_rng(5_000_000 + i)
        ax = crng.normal(size=3); ax /= np.linalg.norm(ax) + 1e-9
        xc = xyz @ Rso3(torch.tensor(ax * crng.uniform(0.5, np.pi), device="cuda", dtype=torch.float32)).T
        res["corrupt"].append(shape_miou(mask_argmax(logits(xc, cls), cls), gt, cls))
        # CanDO: matched-family random pool over SO(3) + identity prior, maximize confidence
        prng = np.random.default_rng(200_000 + i)
        cands = [torch.zeros(3, device="cuda")]
        for _ in range(args.M - 1):
            a = prng.normal(size=3); a /= np.linalg.norm(a) + 1e-9
            cands.append(torch.tensor(a * prng.uniform(0, np.pi), device="cuda", dtype=torch.float32))
        best_v, best_s = cands[0], -1e9
        for v in cands:
            score = conf(xc @ Rso3(v).T, cls) - LAM * float(v.norm())
            if score > best_s:
                best_s, best_v = score, v
        res["recovered"].append(shape_miou(mask_argmax(logits(xc @ Rso3(best_v).T, cls), cls), gt, cls))
        if (j + 1) % 64 == 0:
            print(f"  {j+1}/{len(idx)}", flush=True)

    m = {k: float(np.mean(v)) for k, v in res.items()}
    print(f"[VN part-seg no-op] model={args.model} n={len(idx)}")
    print(f"  clean {m['clean']:.3f} | corrupt(SO3) {m['corrupt']:.3f} | +CanDO {m['recovered']:.3f}")


if __name__ == "__main__":
    main()
