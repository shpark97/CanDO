"""ModelNet40 classification dataset (modelnet40_normal_resampled format) — minimal test-split reader.

Each shape is a .txt of 10000 points (x,y,z,nx,ny,nz). We FPS to npoints and unit-sphere normalize,
matching yanx27's ModelNetDataLoader. Used for the classification experiments.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from torch_cluster import fps


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    pc = pc - pc.mean(axis=0, keepdims=True)
    m = np.sqrt((pc ** 2).sum(axis=1)).max()
    return pc / (m + 1e-9)


class ModelNet40Cls:
    def __init__(self, root: str, split: str = "test", npoints: int = 1024):
        self.root = root
        self.npoints = npoints
        self.cats = [l.strip() for l in open(os.path.join(root, "modelnet40_shape_names.txt"))]
        self.cat2id = {c: i for i, c in enumerate(self.cats)}
        ids = [l.strip() for l in open(os.path.join(root, f"modelnet40_{split}.txt"))]
        self.items = []
        for sid in ids:
            cat = "_".join(sid.split("_")[:-1])
            self.items.append((cat, os.path.join(root, cat, sid + ".txt")))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int, device: str = "cuda") -> dict:
        cat, path = self.items[i]
        pts = np.loadtxt(path, delimiter=",").astype(np.float32)  # (10000, 6)
        xyz = torch.from_numpy(pts[:, :3]).to(device)
        if xyz.shape[0] > self.npoints:
            idx = fps(xyz, ratio=self.npoints / xyz.shape[0], random_start=False)[: self.npoints]
            xyz = xyz[idx]
        xyz = torch.from_numpy(pc_normalize(xyz.cpu().numpy())).to(device)
        return {"points": xyz, "label": self.cat2id[cat], "cat": cat}
