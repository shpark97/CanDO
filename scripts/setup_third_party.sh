#!/usr/bin/env bash
# Clone the third-party code bases used by the experiments at the exact commits we used, and apply our
# local patches. Run once from the repo root:  bash scripts/setup_third_party.sh
#
#   Pointnet_Pointnet2_pytorch  frozen PointNet / PointNet++ models + their released checkpoints (log/)
#   PointNeXt (+ openpoints)    frozen PointNeXt-S models (needs CUDA ops, see README)
#   vnn                         Vector Neurons (equivariant part-segmentation models)
#   PointCloud-C                official ModelNet-C / ShapeNet-C corruption functions
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party

clone() {  # $1=dir $2=url $3=commit
  local dir=third_party/$1
  if [ ! -d "$dir/.git" ]; then
    git clone "$2" "$dir"
  fi
  git -C "$dir" checkout -q "$3"
  echo "[ok] $1 @ ${3:0:7}"
}

clone Pointnet_Pointnet2_pytorch https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git eb64fe0b4c24055559cea26299cb485dcb43d8dd
clone PointNeXt                  https://github.com/guochengqian/PointNeXt.git             8a0ec91de5329b3b3e346422fa43a51ba24fd450
clone vnn                        https://github.com/FlyingGiraffe/vnn.git                  f3197e210022b5f0015e0da6456adf66bd0cd73e
clone PointCloud-C               https://github.com/ldkong1205/PointCloud-C.git            f00a98bea9bb1441fe6b354f853b3e85f756b02b

# openpoints is a submodule of PointNeXt registered with an SSH URL; fetch it over HTTPS instead.
if [ ! -f third_party/PointNeXt/openpoints/__init__.py ]; then
  git -C third_party/PointNeXt config submodule.openpoints.url https://github.com/guochengqian/openpoints.git
  git -C third_party/PointNeXt submodule update --init openpoints
fi
git -C third_party/PointNeXt/openpoints checkout -q baeca5e319aa2e756d179e494469eb7f5ffd29f0
echo "[ok] openpoints @ baeca5e"

# vnn: make pytorch3d optional and allow multi-GPU part-segmentation training.
if git -C third_party/vnn apply --check ../../patches/vnn.patch 2>/dev/null; then
  git -C third_party/vnn apply ../../patches/vnn.patch
  echo "[ok] applied patches/vnn.patch"
else
  echo "[skip] patches/vnn.patch already applied"
fi
