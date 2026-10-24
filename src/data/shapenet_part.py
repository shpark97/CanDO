"""ShapeNetPart part-segmentation dataset (v0_normal format).

Source: shapenetcore_partanno_segmentation_benchmark_v0_normal (see README, "Datasets").
Each shape is a .txt with rows `x y z nx ny nz part_label`, where `part_label` is the
**global** part id (0-49). Category id (0-15) is derived from the synset folder.

This loader keeps the **full** (variable-size) point cloud by default; predictions are evaluated on
every point. Pass `npoints` to draw a fixed number of points (random or fps) when fixed-size batching
is needed.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

# synset id -> (category name, category id 0-15, global part-id range inclusive)
# order matches synsetoffset2category.txt and the SAMBLE mapping.
CATEGORIES = [
    ("02691156", "Airplane", 0, (0, 3)),
    ("02773838", "Bag", 1, (4, 5)),
    ("02954340", "Cap", 2, (6, 7)),
    ("02958343", "Car", 3, (8, 11)),
    ("03001627", "Chair", 4, (12, 15)),
    ("03261776", "Earphone", 5, (16, 18)),
    ("03467517", "Guitar", 6, (19, 21)),
    ("03624134", "Knife", 7, (22, 23)),
    ("03636649", "Lamp", 8, (24, 27)),
    ("03642806", "Laptop", 9, (28, 29)),
    ("03790512", "Motorbike", 10, (30, 35)),
    ("03797390", "Mug", 11, (36, 37)),
    ("03948459", "Pistol", 12, (38, 40)),
    ("04099429", "Rocket", 13, (41, 43)),
    ("04225987", "Skateboard", 14, (44, 46)),
    ("04379243", "Table", 15, (47, 49)),
]
SYNSET_TO_CLS = {s: c for s, _, c, _ in CATEGORIES}
NUM_CATEGORIES = 16
NUM_PARTS = 50


def normalize_unit_sphere(xyz: np.ndarray) -> np.ndarray:
    """Center to centroid and scale to the unit sphere (standard ShapeNetPart preprocessing)."""
    xyz = xyz - xyz.mean(axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(xyz, axis=1))
    if scale > 0:
        xyz = xyz / scale
    return xyz


class ShapeNetPart(Dataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        npoints: Optional[int] = None,
        normalize: bool = True,
        with_normals: bool = True,
        cache: bool = True,
    ):
        assert split in ("train", "val", "test", "trainval")
        self.root = root
        self.split = split
        self.npoints = npoints
        self.normalize = normalize
        self.with_normals = with_normals
        self.cache = cache
        self._cache_dir = os.path.join(root, "processed", "npy")
        if cache:
            os.makedirs(self._cache_dir, exist_ok=True)

        splits = ["train", "val"] if split == "trainval" else [split]
        self.files: list[tuple[str, int]] = []  # (txt_path, cls_id)
        for sp in splits:
            jpath = os.path.join(root, "train_test_split", f"shuffled_{sp}_file_list.json")
            for entry in json.load(open(jpath)):
                synset, shape_id = entry.split("/")[-2:]
                txt = os.path.join(root, synset, shape_id + ".txt")
                self.files.append((txt, SYNSET_TO_CLS[synset]))

    def __len__(self) -> int:
        return len(self.files)

    def _load_raw(self, txt: str) -> np.ndarray:
        """Return (N, 7) float array, with .npy caching for speed (txt parsing is slow)."""
        if self.cache:
            key = os.path.relpath(txt, self.root).replace(os.sep, "__").replace(".txt", ".npy")
            npy = os.path.join(self._cache_dir, key)
            if os.path.exists(npy):
                return np.load(npy)
            arr = np.loadtxt(txt).astype(np.float32)
            np.save(npy, arr)
            return arr
        return np.loadtxt(txt).astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        txt, cls_id = self.files[idx]
        arr = self._load_raw(txt)
        xyz, normals, seg = arr[:, 0:3], arr[:, 3:6], arr[:, 6].astype(np.int64)

        if self.npoints is not None:
            n = xyz.shape[0]
            replace = n < self.npoints
            sel = np.random.choice(n, self.npoints, replace=replace)
            xyz, normals, seg = xyz[sel], normals[sel], seg[sel]

        if self.normalize:
            xyz = normalize_unit_sphere(xyz)

        out = {
            "points": torch.from_numpy(xyz).float(),       # (N, 3)
            "seg": torch.from_numpy(seg).long(),           # (N,) global part id 0-49
            "cls": torch.tensor(cls_id, dtype=torch.long),  # () category id 0-15
        }
        if self.with_normals:
            out["normals"] = torch.from_numpy(normals).float()  # (N, 3)
        return out
