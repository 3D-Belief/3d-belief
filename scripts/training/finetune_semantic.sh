#!/bin/bash
set -e

# ---- environment ----
eval "$(conda shell.bash hook)"
conda activate 3d-belief

nvidia-smi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Ensure local repo is found before any pip-installed copy
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

export MASTER_PORT=$((12000 + RANDOM % 1000))

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
NGPUS="${1:-${NGPUS:-1}}"
# wandb mode (online|local) and entity: overridable via 2nd/3rd CLI arg or env var
WANDB="${2:-${WANDB:-local}}"
WANDB_ENTITY="${3:-${WANDB_ENTITY:-null}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPUS - 1)))}"
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/spoc}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/3d_belief_spoc.pt}"

CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes 1 --nproc_per_node $NGPUS --master_port $MASTER_PORT\
    splat_belief/experiment/train.py \
    dataset=spoc \
    dataset.vggt_alignment_loss_weight=2.0 \
    dataset.intermediate_weight=5.0 \
    dataset.depth_smooth_loss_weight=0.1 \
    dataset.depth_loss_weight=1.0 \
    dataset.root_dir="${DATASET_ROOT}" \
    setting_name=pixelsplat_h100 \
    stage=train \
    results_folder=outputs/training/spoc_semantic \
    semantic_config=${REPO_ROOT}/splat_belief/config/semantic/onehot.yaml \
    checkpoint_path="${CHECKPOINT_PATH}" \
    ngpus=$NGPUS \
    image_size=128 \
    ctxt_min=5 \
    ctxt_max=15 \
    model/encoder=uvitmvsplat \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=true \
    model.encoder.use_reg_model=true \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=768 \
    model.encoder.reg_model_name=dinov3_base \
    model.encoder.reg_model_weights=${REPO_ROOT}/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_depth_mask=true \
    model.encoder.freeze_depth_predictor=false \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.use_vggt_alignment=false \
    model.encoder.backbone.use_repa=false \
    model.encoder.backbone.input_size='[128, 128]' \
    alignment.latents_info=-1 \
    ctxt_losses_factor=0.7 \
    repa_encoder_name=dinov3-vit-b \
    repa_encoder_weights=${REPO_ROOT}/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    name=spoc_semantic \
    wandb=$WANDB \
    ++wandb.entity=$WANDB_ENTITY \
    clean_target=false \
    use_identity=true \
    intermediate=true \
    load_optimizer=false \
    load_enc=false \
    finetune_component=semantic_head \
    finetune_steps=500000000 \
    use_depth_smoothness=true \
    adjacent_angle=0.785 \
    adjacent_distance=1.0 \
    num_intermediate=15
