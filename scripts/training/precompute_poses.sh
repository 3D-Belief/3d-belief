#!/bin/bash
# Pre-compute CameraHead poses for ProcTHOR dataset.
# Wrapper for splat_belief/experiment/precompute_poses.py
set -e

# ---- paths ----
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${REPO_ROOT}/../datasets/poc_dataset"
# Set to a finetuned checkpoint path, or leave empty to use pretrained VGGT CameraHead
CAMERA_HEAD_CKPT="${REPO_ROOT}/checkpoints/camera_head_procthor/camera_head_best.pth"

# ---- environment ----
eval "$(conda shell.bash hook)"
conda activate 3d-belief

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

cd "${REPO_ROOT}"

CKPT_ARGS=""
if [ -n "${CAMERA_HEAD_CKPT}" ] && [ -f "${CAMERA_HEAD_CKPT}" ]; then
    CKPT_ARGS="--camera_head_ckpt ${CAMERA_HEAD_CKPT}"
fi

python splat_belief/experiment/precompute_poses.py \
    --data_root "${DATASET_ROOT}" \
    ${CKPT_ARGS} \
    --stage train \
    --batch_frames 16 \
    "$@"
