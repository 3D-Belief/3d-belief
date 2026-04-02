#!/bin/bash
# Finetune VGGT CameraHead on ProcTHOR dataset.
# Wrapper for splat_belief/experiment/finetune_camera_head.py
set -e

# ---- paths ----
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${REPO_ROOT}/../datasets/poc_dataset"

# ---- environment ----
eval "$(conda shell.bash hook)"
conda activate 3d-belief

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0

cd "${REPO_ROOT}"

python splat_belief/experiment/finetune_camera_head.py \
    --data_root "${DATASET_ROOT}" \
    --output_dir checkpoints/camera_head_procthor \
    --epochs 20 \
    --lr 1e-4 \
    --batch_size 4 \
    --num_views 8 \
    "$@"
