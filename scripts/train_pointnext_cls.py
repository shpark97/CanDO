"""Train a frozen-target PointNeXt-S ModelNet40 classifier on modelnet40_normal_resampled.

SAME data + SAME unit-sphere normalization as src.data.ModelNet40Cls (so the SO(3) recovery harness
canon_modelnet feeds identical points), and NO rotation augmentation (pose-sensitive, like our
PointNet++ classifier), so an SO(3) corruption leaves recoverable headroom. Caches FPS'd+normalized
points once, then trains with adamw + cosine + label smoothing + scale/translate aug.

Run:  PYTHONPATH=third_party/PointNeXt:. python scripts/train_pointnext_cls.py --epochs 300
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_cluster import fps

from src.data.modelnet import pc_normalize
from src.downstream.pointnext_cls import build_pointnext_cls

ROOT = "datasets/modelnet_dl"
CACHE = "datasets/modelnet_dl/pnx_cls_1024_cache.pt"
DEFAULT_CFG = "third_party/PointNeXt/cfgs/modelnet40ply2048/pointnext-s.yaml"
DEFAULT_OUT = "pretrained/pointnext_modelnet/best_model.pth"


def build_cache(npoints=1024, device="cuda"):
    if os.path.exists(CACHE):
        return torch.load(CACHE)
    cats = [l.strip() for l in open(os.path.join(ROOT, "modelnet40_shape_names.txt"))]
    cat2id = {c: i for i, c in enumerate(cats)}
    out = {}
    for split in ("train", "test"):
        ids = [l.strip() for l in open(os.path.join(ROOT, f"modelnet40_{split}.txt"))]
        X, Y = [], []
        for k, sid in enumerate(ids):
            cat = "_".join(sid.split("_")[:-1])
            arr = pd.read_csv(os.path.join(ROOT, cat, sid + ".txt"), header=None).values[:, :3].astype(np.float32)
            xyz = torch.from_numpy(arr).to(device)
            if xyz.shape[0] > npoints:
                idx = fps(xyz, ratio=npoints / xyz.shape[0], random_start=False)[:npoints]
                xyz = xyz[idx]
            xyz = pc_normalize(xyz.cpu().numpy())
            X.append(xyz); Y.append(cat2id[cat])
            if (k + 1) % 2000 == 0:
                print(f"  cache {split} {k+1}/{len(ids)}", flush=True)
        out[f"{split}_x"] = torch.from_numpy(np.stack(X)).float()
        out[f"{split}_y"] = torch.tensor(Y, dtype=torch.long)
        print(f"  cached {split}: {out[f'{split}_x'].shape}", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    torch.save(out, CACHE)
    return out


def scale_translate(x, smin=0.9, smax=1.1, shift=0.2):
    """PointCloudScaleAndTranslate: per-sample anisotropic scale + global shift (no rotation)."""
    B = x.shape[0]
    s = torch.empty(B, 1, 3, device=x.device).uniform_(smin, smax)
    t = torch.empty(B, 1, 3, device=x.device).uniform_(-shift, shift)
    return x * s + t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ls", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg", default=DEFAULT_CFG, help="openpoints cls config (pointnext-s / pointvector-s / pointmlp)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    data = build_cache()
    Xtr, Ytr = data["train_x"].to(dev), data["train_y"].to(dev)
    Xte, Yte = data["test_x"].to(dev), data["test_y"].to(dev)
    ntr = Xtr.shape[0]
    print(f"train {ntr}  test {Xte.shape[0]}", flush=True)

    net = build_pointnext_cls(cfg_path=args.cfg, device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=args.lr * 1e-2)

    best = 0.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ep in range(args.epochs):
        net.train()
        perm = torch.randperm(ntr, device=dev)
        for i in range(0, ntr, args.bs):
            bidx = perm[i:i + args.bs]
            xb = scale_translate(Xtr[bidx]); yb = Ytr[bidx]
            opt.zero_grad()
            logits = net(xb.contiguous())
            loss = F.cross_entropy(logits, yb, label_smoothing=args.ls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        # eval (instance accuracy, no aug)
        net.eval(); correct = 0
        with torch.no_grad():
            for i in range(0, Xte.shape[0], 64):
                logits = net(Xte[i:i + 64].contiguous())
                correct += (logits.argmax(-1) == Yte[i:i + 64]).sum().item()
        acc = correct / Xte.shape[0]
        if acc > best:
            best = acc
            torch.save({"model_state_dict": net.state_dict(), "acc": best, "epoch": ep}, args.out)
        if (ep + 1) % 5 == 0 or ep < 3:
            print(f"epoch {ep+1}/{args.epochs}  test_acc {acc:.4f}  best {best:.4f}", flush=True)
    print(f"DONE best_test_acc {best:.4f} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
