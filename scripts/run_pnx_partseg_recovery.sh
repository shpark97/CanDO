#!/usr/bin/env bash
# PointNeXt-S ShapeNetPart pose recovery (azimuth + SO(3)) x 3 corruption seeds on the full test set
# (Table 1, PointNeXt-S ShapeNetPart rows). Shards the test set over NGPU GPUs; per-shape randomness is
# keyed by the global index, so any sharding reproduces the single-process numbers.
#
# Usage (from the repo root):  NGPU=8 bash scripts/run_pnx_partseg_recovery.sh
set -u
PY=${PY:-python}
NGPU=${NGPU:-1}
CKPT=${CKPT:-pretrained/pointnext_shapenetpart/best.pth}
echo "CKPT=$CKPT NGPU=$NGPU"
N=$(PYTHONPATH=. $PY -c "from src.data.shapenet_part import ShapeNetPart; print(len(ShapeNetPart('datasets/shapenetcore_partanno_segmentation_benchmark_v0_normal','test',npoints=None,normalize=False)))")
S=$(( (N + NGPU - 1) / NGPU ))
echo "N=$N S=$S"
mkdir -p runs/pnx_ps
for T in azimuth so3; do
  for C in 0 1 2; do
    for (( g = 0; g < NGPU; g++ )); do
      a=$(( g * S )); b=$(( (g + 1) * S ))
      CUDA_VISIBLE_DEVICES=$g PYTHONPATH=third_party/PointNeXt:. $PY -m src.eval.canon_partseg_pnx \
        --transform "$T" --n -1 --cseed "$C" --idx_start "$a" --idx_end "$b" --ckpt "$CKPT" \
        --dump runs/pnx_ps/${T}_c${C}_g${g}.json > runs/pnx_ps/${T}_c${C}_g${g}.out 2>&1 &
    done
    wait
    echo "DONE transform=$T cseed=$C"
  done
done
echo "ALL_RECOVERY_DONE"
