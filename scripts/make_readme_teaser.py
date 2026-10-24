"""Render the README teaser GIF (docs/static/images/teaser.gif) from the project-page examples.

Each example spins in three synced panels: the misoriented input with the frozen model's parts, the
CanDO-canonicalized input with its parts, and the upright ground truth. Uses the same orthographic,
depth-faded rendering and colors as the interactive viewer in docs/index.html.

Run:  python scripts/make_readme_teaser.py
"""
from __future__ import annotations

import argparse
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

PALETTE = np.array([[31, 119, 180], [214, 39, 40], [23, 190, 207], [227, 119, 194], [140, 86, 75], [255, 127, 14]]) / 255
GREEN, RED, GRAY = "#1f9d55", "#c0392b", "#5b6168"


def load(entry, scale):
    xyz = np.asarray(entry["xyz"], dtype=np.float32).reshape(-1, 3)
    return (xyz - xyz.mean(0)) / scale, np.asarray(entry["part"])


def project(xyz, yaw, pitch):
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    x1 = cy * xyz[:, 0] + sy * xyz[:, 2]; z1 = -sy * xyz[:, 0] + cy * xyz[:, 2]
    y2 = cp * xyz[:, 1] - sp * z1; z2 = sp * xyz[:, 1] + cp * z1
    return x1, y2, z2


def frame(shape, clouds, yaw, pitch=-0.35, size=(9.6, 3.6), dpi=100):
    fig, axes = plt.subplots(1, 3, figsize=size, dpi=dpi)
    fig.patch.set_facecolor("white")
    titles = [("Misoriented input", f"mIoU {shape['miou_source']:.2f}", RED),
              ("After CanDO", f"mIoU {shape['miou_cando']:.2f}", GREEN),
              ("Upright ground truth", "reference", GRAY)]
    for ax, (xyz, part), (t, sub, col) in zip(axes, clouds, titles):
        x, y, z = project(xyz, yaw, pitch)
        o = np.argsort(z)
        f = (0.55 + 0.45 * (z[o] + 1) / 2)[:, None]
        c = 1 - (1 - PALETTE[part[o] % len(PALETTE)]) * f                 # fade far points toward white
        ax.scatter(x[o], y[o], c=c, s=3.6, linewidths=0)
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.0, 1.1); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(t, fontsize=12, fontweight="bold", color="#16181a", pad=2)
        ax.text(0.5, -0.02, sub, transform=ax.transAxes, ha="center", va="top", fontsize=11, color=col, fontweight="bold")
    fig.text(0.5, 0.955, f"{shape['category']}  ·  rotated {shape['angle_deg']:.0f}° about a random axis  ·  frozen PointNet++",
             ha="center", fontsize=10, color=GRAY)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.08, wspace=0.02)
    buf = io.BytesIO(); fig.savefig(buf, format="png", facecolor="white"); plt.close(fig)
    return Image.open(buf).convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="docs/static/data/shapes.json")
    ap.add_argument("--out", default="docs/static/images/teaser.gif")
    ap.add_argument("--shapes", nargs="+", default=["Airplane", "Chair", "Table", "Skateboard"])
    ap.add_argument("--frames", type=int, default=30, help="frames per shape (one full turn)")
    args = ap.parse_args()
    shapes = {s["category"]: s for s in json.load(open(args.data))["shapes"]}
    frames = []
    for name in args.shapes:
        s = shapes[name]
        t = np.asarray(s["target"]["xyz"], dtype=np.float32).reshape(-1, 3)
        scale = np.linalg.norm(t - t.mean(0), axis=1).max()
        clouds = [load(s[k], scale) for k in ("input", "cando", "target")]
        for k in range(args.frames):
            frames.append(frame(s, clouds, 0.6 + 2 * np.pi * k / args.frames))
    pal = frames[0].quantize(colors=128, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(args.out, save_all=True, append_images=q[1:], duration=90, loop=0, optimize=True)
    print(f"wrote {len(frames)} frames -> {args.out}")


if __name__ == "__main__":
    main()
