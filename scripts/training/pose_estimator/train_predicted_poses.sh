#!/bin/bash
# Finetune 3d-belief using pre-computed CameraHead predicted poses.
#
# Prerequisites:
# 1. Finetune CameraHead: bash scripts/training/finetune_camera_head.sh
# 2. Pre-compute poses:   bash scripts/training/precompute_poses.sh
# 3. Run this script to finetune 3d-belief on predicted poses.
set -e

# ---- paths ----
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${REPO_ROOT}/../datasets/poc_dataset"
VOCAB_DIR="${REPO_ROOT}/outputs/vocab/procthor"

# ---- environment ----
eval "$(conda shell.bash hook)"
conda activate 3d-belief

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=0
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export MASTER_PORT=$((12000 + RANDOM % 1000))

# ---- training ----
CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port $MASTER_PORT \
    splat_belief/experiment/train.py \
    dataset=procthor \
    dataset.root_dir="${DATASET_ROOT}" \
    dataset.vocab_dir="${VOCAB_DIR}" \
    dataset.pose_source=predicted \
    dataset.predicted_poses_filename=predicted_poses.npz \
    dataset.vggt_alignment_loss_weight=2.0 \
    dataset.intermediate_weight=5.0 \
    dataset.depth_smooth_loss_weight=0.1 \
    setting_name=predicted_poses \
    stage=train \
    results_folder=outputs/training/procthor_predicted_poses \
    semantic_config=splat_belief/config/semantic/onehot.yaml \
    checkpoint_path=outputs/training/procthor_base_weights/model-5.pt \
    pose_source=predicted \
    ngpus=1 \
    image_size=128 \
    ctxt_min=5 \
    ctxt_max=15 \
    model/encoder=uvitmvsplat \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=false \
    model.encoder.use_reg_model=false \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=384 \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_depth_mask=true \
    model.encoder.encoder_ckpt=checkpoints/re10k.ckpt \
    model.encoder.freeze_depth_predictor=false \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.use_vggt_alignment=true \
    model.encoder.backbone.use_repa=true \
    model.encoder.backbone.input_size='[128, 128]' \
    alignment.latents_info=-1 \
    ctxt_losses_factor=0.9 \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    name=procthor_predicted_poses \
    wandb=local \
    clean_target=false \
    use_identity=true \
    intermediate=true \
    load_optimizer=false \
    load_enc=true \
    lock_enc_steps=5 \
    use_depth_smoothness=true \
    adjacent_angle=0.785 \
    adjacent_distance=1.0 \
    num_intermediate=15
