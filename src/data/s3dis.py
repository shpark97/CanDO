"""S3DIS scene dataset (Pointcept format) + 1m×1m block sampling matching yanx27 pointnet2_sem_seg.

Per room: coord/color/normal/segment .npy (~300k pts). 13 classes. Area_5 = test (standard).
The frozen yanx27 sem_seg expects 4096-pt 1m×1m blocks with a 9-channel feature:
    [x-blockctr_x, y-blockctr_y, z, r/255, g/255, b/255, x/room_max, y/room_max, z/room_max]
Note: Pointcept coords are NOT room-min-subtracted, so we subtract per-room min (yanx27 assumption).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import torch

NUM_CLASSES = 13
CLASS_NAMES = ["ceiling", "floor", "wall", "beam", "column", "window", "door",
               "table", "chair", "sofa", "bookcase", "board", "clutter"]


class S3DISRooms:
    def __init__(self, root: str, split: str = "test", test_area: int = 5):
        self.root = root
        rooms = sorted(glob.glob(os.path.join(root, "Area_*", "*")))
        rooms = [r for r in rooms if os.path.isdir(r)]
        tag = f"Area_{test_area}"
        if split == "train":
            self.rooms = [r for r in rooms if tag not in r]
        else:
            self.rooms = [r for r in rooms if tag in r]

    def __len__(self):
        return len(self.rooms)

    def load(self, idx: int, device: str = "cuda") -> dict:
        d = self.rooms[idx]
        coord = np.load(os.path.join(d, "coord.npy")).astype(np.float32)
        coord = coord - coord.min(axis=0, keepdims=True)          # room-min subtract (yanx27)
        color = np.load(os.path.join(d, "color.npy")).astype(np.float32)
        seg = np.load(os.path.join(d, "segment.npy")).reshape(-1).astype(np.int64)
        return {
            "coord": torch.from_numpy(coord).to(device),         # (N,3) >= 0
            "color": torch.from_numpy(color).to(device),         # (N,3) 0-255
            "seg": torch.from_numpy(seg).to(device),             # (N,) 0-12
            "room_max": torch.from_numpy(coord.max(axis=0)).to(device),  # (3,)
            "name": os.path.relpath(d, self.root),
        }


def sample_block(room: dict, block_size: float = 1.0, min_points: int = 1024,
                 generator: torch.Generator | None = None, max_tries: int = 20) -> dict | None:
    """Pick a random 1m×1m (x,y) column block with > min_points. Returns the block's FULL points
    (variable size = the 'dense cloud' to subsample from), or None if none found."""
    coord = room["coord"]
    n = coord.shape[0]
    for _ in range(max_tries):
        c = coord[torch.randint(n, (1,), device=coord.device, generator=generator)][0]
        lo = c - torch.tensor([block_size / 2, block_size / 2, 1e9], device=coord.device)
        hi = c + torch.tensor([block_size / 2, block_size / 2, 1e9], device=coord.device)
        m = ((coord[:, 0] >= lo[0]) & (coord[:, 0] <= hi[0]) &
             (coord[:, 1] >= lo[1]) & (coord[:, 1] <= hi[1]))
        idx = m.nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > min_points:
            return {"idx": idx, "center": c, "coord": coord[idx], "color": room["color"][idx],
                    "seg": room["seg"][idx], "room_max": room["room_max"]}
    return None


def make_semseg_input(block: dict, sel: torch.Tensor) -> torch.Tensor:
    """Build (1, 9, K) input for yanx27 sem_seg from selected indices `sel` into the block."""
    coord, color, center, rmax = block["coord"][sel], block["color"][sel], block["center"], block["room_max"]
    feat = torch.empty((coord.shape[0], 9), device=coord.device)
    feat[:, 0] = coord[:, 0] - center[0]
    feat[:, 1] = coord[:, 1] - center[1]
    feat[:, 2] = coord[:, 2]
    feat[:, 3:6] = color / 255.0
    feat[:, 6] = coord[:, 0] / (rmax[0] + 1e-8)
    feat[:, 7] = coord[:, 1] / (rmax[1] + 1e-8)
    feat[:, 8] = coord[:, 2] / (rmax[2] + 1e-8)
    return feat.t().unsqueeze(0).contiguous()  # (1, 9, K)
