"""Controlled augmentation ablation (makes the augmentation-gap claim causal).

Train the SAME PointNet++ SSG classifier on ModelNet40 four times, varying ONLY the rotation augmentation
regime: none / yaw (azimuth) / tilt (gravity-axis) / so3 (full). Each checkpoint is then probed under
matched corruptions. The principle predicts a transform is recoverable only when it
lies in the model's augmentation gap (so the model is sensitive) and the confidence surrogate stays aligned.

Run:  PYTHONPATH=. python -m src.eval.train_aug_ablation --rot_aug yaw --epochs 120 \
          --out third_party/Pointnet_Pointnet2_pytorch/log/aug_ablation/yaw/checkpoints/best_model.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

from src.data.modelnet import ModelNet40Cls

# make the yanx27 `models` package importable (mirrors FrozenModelNetCls)
_PN2 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                    "third_party", "Pointnet_Pointnet2_pytorch"))
for _p in (_PN2, os.path.join(_PN2, "models")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def rand_rot(B, regime, device, rng, amin=0.0):
    """(B,3,3) batched rotation matrices for the augmentation/corruption regime.

    ``amin`` floors the rotation angle (radians) for the tilt/so3 families; the default 0.0 preserves the
    original training/augmentation behaviour. Set ``amin>0`` to match the deployed-headline corruption
    convention, e.g. ``amin=0.5`` for so3 angle ~ U(0.5, pi) as in baseline_memo."""
    if regime == "none":
        return torch.eye(3, device=device).expand(B, 3, 3).contiguous()
    if regime == "yaw":
        a = torch.tensor(rng.uniform(0, 2 * np.pi, size=B), device=device, dtype=torch.float32)
        axis = torch.zeros(B, 3, device=device); axis[:, 2] = 1.0
    elif regime == "tilt":
        phi = torch.tensor(rng.uniform(0, 2 * np.pi, size=B), device=device, dtype=torch.float32)
        a = torch.tensor(rng.uniform(amin, np.pi, size=B), device=device, dtype=torch.float32)
        axis = torch.stack([torch.cos(phi), torch.sin(phi), torch.zeros_like(phi)], -1)
    else:  # so3
        axis = torch.tensor(rng.normal(size=(B, 3)), device=device, dtype=torch.float32)
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-9)
        a = torch.tensor(rng.uniform(amin, np.pi, size=B), device=device, dtype=torch.float32)
    # Rodrigues, batched
    k = axis
    K = torch.zeros(B, 3, 3, device=device)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    I = torch.eye(3, device=device).expand(B, 3, 3)
    s = torch.sin(a).view(B, 1, 1); c = (1 - torch.cos(a)).view(B, 1, 1)
    return I + s * K + c * (K @ K)


def preload(ds):
    pts, lab = [], []
    for i in range(len(ds)):
        s = ds[i]; pts.append(s["points"].float().cpu()); lab.append(int(s["label"]))
        if (i + 1) % 2000 == 0:
            print(f"  preload {i+1}/{len(ds)}", flush=True)
    return torch.stack(pts), torch.tensor(lab, dtype=torch.long)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/modelnet_dl")
    ap.add_argument("--rot_aug", choices=["none", "yaw", "tilt", "so3"], required=True)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--npoints", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    from models.pointnet2_cls_ssg import get_model, get_loss
    net = get_model(40, normal_channel=False).cuda()
    crit = get_loss().cuda()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=20, gamma=0.7)

    print(f"[train aug={args.rot_aug}] preloading train set...", flush=True)
    Xtr, Ytr = preload(ModelNet40Cls(args.root, "train", args.npoints))
    print(f"  train: {len(Xtr)} samples", flush=True)
    Xte, Yte = preload(ModelNet40Cls(args.root, "test", args.npoints))

    n = len(Xtr); best = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        net.train(); perm = rng.permutation(n)
        for s in range(0, n, args.bs):
            bi = perm[s:s + args.bs]
            pts = Xtr[bi].cuda(); y = Ytr[bi].cuda()
            R = rand_rot(len(bi), args.rot_aug, "cuda", rng)
            pts = torch.bmm(pts, R.transpose(1, 2))                 # rotate
            pts = pts - pts.mean(1, keepdim=True)                   # recenter
            pred, tf = net(pts.transpose(1, 2).contiguous())
            loss = crit(pred, y, tf)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            net.eval(); correct = 0
            with torch.no_grad():
                for s in range(0, len(Xte), 64):
                    pts = Xte[s:s + 64].cuda()
                    pred, _ = net(pts.transpose(1, 2).contiguous())
                    correct += (pred.argmax(1).cpu() == Yte[s:s + 64]).sum().item()
            acc = correct / len(Xte)
            print(f"  epoch {ep+1}/{args.epochs}  clean-test acc={acc:.4f}", flush=True)
            if acc >= best:
                best = acc
                torch.save({"model_state_dict": net.state_dict(), "acc": acc, "rot_aug": args.rot_aug}, args.out)
    print(f"[train aug={args.rot_aug}] best clean-test acc={best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
