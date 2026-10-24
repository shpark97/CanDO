"""Controlled augmentation ablation on PART SEGMENTATION.

Trains the same PointNet++ MSG part segmentation network under rotation-augmentation regimes
(none / yaw / tilt / so3) so the augmentation-gap principle gets a causal test on the task where the
surrogate is strongest (Table 4). Shares the rotation regimes of src/eval/train_aug_ablation.py.

Run (one GPU per regime):
  PYTHONPATH=. python -m src.eval.train_partseg_aug --rot_aug none --epochs 100 \
      --out pretrained/pointnet2_partseg_aug/none/best_model.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from src.data.shapenet_part import ShapeNetPart
from src.downstream.pointnet2_partseg import NUM_CATEGORIES, _PN2_ROOT
from src.eval.metrics import PART_RANGES, mask_argmax, shape_miou
from src.eval.train_aug_ablation import rand_rot  # shared none/yaw/tilt/so3 regime (matches the eval side)


def preload(ds, npoints, rng):
    """Preload to tensors with a fixed random subsample per shape."""
    X, N, S, C = [], [], [], []
    for i in range(len(ds)):
        s = ds[i]
        pts, nrm, seg = s["points"], s["normals"], s["seg"]
        if pts.shape[0] >= npoints:
            sel = rng.choice(pts.shape[0], npoints, replace=False)
        else:
            sel = rng.choice(pts.shape[0], npoints, replace=True)
        X.append(pts[sel]); N.append(nrm[sel]); S.append(seg[sel]); C.append(s["cls"])
    return (torch.stack(X), torch.stack(N), torch.stack(S).long(), torch.stack(C).long())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal")
    ap.add_argument("--rot_aug", choices=["none", "yaw", "tilt", "so3"], required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--npoints", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_eval", type=int, default=256, help="test subset for the periodic clean check")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    if os.path.abspath(_PN2_ROOT) not in sys.path:
        sys.path.insert(0, os.path.abspath(_PN2_ROOT))
    from models.pointnet2_part_seg_msg import get_model, get_loss
    net = get_model(50, normal_channel=True).cuda()
    crit = get_loss().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.5)

    print(f"[train partseg aug={args.rot_aug}] preloading train set...", flush=True)
    Xtr, Ntr, Str, Ctr = preload(ShapeNetPart(args.root, "trainval"), args.npoints, rng)
    print(f"  train: {len(Xtr)} shapes", flush=True)
    ds_te = ShapeNetPart(args.root, "test", npoints=None)
    eval_idx = np.random.default_rng(0).choice(len(ds_te), args.n_eval, replace=False).tolist()

    n = len(Xtr); best = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        net.train()
        perm = rng.permutation(n)
        for b0 in range(0, n, args.bs):
            bi = perm[b0:b0 + args.bs]
            xyz = Xtr[bi].cuda(); nrm = Ntr[bi].cuda()
            seg = Str[bi].cuda(); cls = Ctr[bi].cuda()
            R = rand_rot(len(bi), args.rot_aug, "cuda", rng)
            xr = torch.einsum("bij,bnj->bni", R, xyz)
            nr = torch.einsum("bij,bnj->bni", R, nrm)
            oh = F.one_hot(cls, NUM_CATEGORIES).float().cuda()
            feat = torch.cat([xr, nr], -1).transpose(1, 2).contiguous()
            logits, trans = net(feat, oh)                               # (B,N,50) log-prob
            loss = crit(logits.reshape(-1, 50), seg.reshape(-1), trans)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            net.eval()
            mis = []
            with torch.no_grad():
                for i in eval_idx:
                    s = ds_te[i]
                    xyz, nrm = s["points"].cuda(), s["normals"].cuda()
                    seg, cls = s["seg"].cuda(), s["cls"].item()
                    oh1 = F.one_hot(torch.tensor([cls]), NUM_CATEGORIES).float().cuda()
                    feat = torch.cat([xyz, nrm], -1).t().unsqueeze(0).contiguous()
                    lg, _ = net(feat, oh1)
                    mis.append(shape_miou(mask_argmax(lg.squeeze(0), cls), seg, cls))
            mi = float(np.mean(mis))
            print(f"  epoch {ep+1}/{args.epochs}  clean-test mIoU={mi:.4f}", flush=True)
            if mi > best:
                best = mi
                torch.save({"model_state_dict": net.state_dict(), "miou": mi, "rot_aug": args.rot_aug},
                           args.out)
    print(f"[train partseg aug={args.rot_aug}] best clean-test mIoU={best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
