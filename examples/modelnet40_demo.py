"""CanDO on a real frozen classifier: PointNet++ on ModelNet40, used as a score-only black box.

Requires `bash scripts/setup_third_party.sh` (PointNet++ code and checkpoint) and ModelNet40 under
datasets/modelnet_dl (see README). Each test shape is rotated by a random SO(3) rotation, classified as is,
and classified again after CanDO canonicalizes it using only the model's softmax outputs.

Run:  PYTHONPATH=. python examples/modelnet40_demo.py --n 20
"""
import argparse

import numpy as np
import torch

from cando import cando, max_prob_confidence, rodrigues, rotate
from src.data.modelnet import ModelNet40Cls
from src.downstream.modelnet_cls import FrozenModelNetCls

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20, help="number of test shapes")
ap.add_argument("--budget", type=int, default=240, help="forward queries per shape")
args = ap.parse_args()

ds = ModelNet40Cls("datasets/modelnet_dl", "test", 1024)
model = FrozenModelNetCls()                                       # frozen; forward queries only
rng = np.random.default_rng(0)
before = after = 0
for i in rng.choice(len(ds), args.n, replace=False):
    s = ds[int(i)]
    x, y = s["points"].cuda().float(), s["label"]
    axis = rng.normal(size=3); axis /= np.linalg.norm(axis)
    x = rotate(x, rodrigues(torch.tensor(axis * rng.uniform(0.5, np.pi), dtype=torch.float32).cuda()))

    def objective(Rs):                                            # (M,3,3) -> (M,) max-softmax
        return max_prob_confidence(model(rotate(x, Rs)).exp())

    R, _, info = cando(objective, budget=args.budget, seed=int(i))
    p0 = int(model(x[None]).argmax()); p1 = int(model(rotate(x, R)[None]).argmax())
    before += p0 == y; after += p1 == y
    print(f"{s['cat']:>12s}: corrupted -> {ds.cats[p0]:<12s} recovered -> {ds.cats[p1]:<12s} "
          f"({info['queries']} queries)")
print(f"\naccuracy  corrupted {before / args.n:.2f}  ->  CanDO {after / args.n:.2f}")
