#!/bin/bash
# Base model training at 256x256 resolution (same setup as train.sh).
set -e

# ---- environment ----
eval "$(conda shell.bash hook)"
conda activate 3d-belief

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Ensure local repo is found before any pip-installed copy
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
NGPUS="${1:-${NGPUS:-2}}"
# wandb mode (online|local) and entity: overridable via 2nd/3rd CLI arg or env var
WANDB="${2:-${WANDB:-local}}"
WANDB_ENTITY="${3:-${WANDB_ENTITY:-null}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NGPUS - 1)))}"
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export MASTER_PORT=$((12000 + RANDOM % 1000))
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/re10k}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/3d_belief_re10k.pt}"

cd "$REPO_ROOT"

CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes 1 --nproc_per_node $NGPUS --master_port $MASTER_PORT \
    splat_belief/experiment/train.py \
    dataset=realestate \
    dataset.vggt_alignment_loss_weight=2.0 \
    dataset.intermediate_weight=5.0 \
    dataset.depth_loss_weight=0.5 \
    dataset.depth_smooth_loss_weight=0.1 \
    dataset.root_dir="${DATASET_ROOT}" \
    setting_name=debug \
    stage=train \
    results_folder=outputs/training/spoc_base_256_from_128_re10k \
    semantic_config=configurations/semantic/onehot.yaml \
    checkpoint_path="${CHECKPOINT_PATH}" \
    use_depth_supervision=false \
    ngpus=$NGPUS \
    image_size=256 \
    ctxt_min=50 \
    ctxt_max=150 \
    model/encoder=uvitmvsplat \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=false \
    model.encoder.use_reg_model=false \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=768 \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_depth_mask=false \
    model.encoder.freeze_depth_predictor=false \
    model.encoder.grid_sample_disable_cudnn=true \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.use_vggt_alignment=false \
    model.encoder.backbone.use_repa=true \
    model.encoder.backbone.input_size='[256, 256]' \
    alignment.latents_info=-1 \
    ctxt_losses_factor=0.8 \
    repa_encoder_name=dinov3-vit-b \
    repa_encoder_weights=${REPO_ROOT}/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    name=spoc_base_256_from_128_re10k \
    wandb=$WANDB \
    ++wandb.entity=$WANDB_ENTITY \
    clean_target=false \
    use_identity=true \
    intermediate=true \
    load_optimizer=false \
    load_enc=false \
    finetune_component=null \
    finetune_steps=0 \
    use_depth_smoothness=true \
    adjacent_angle=0.785 \
    adjacent_distance=1.0 \
    num_intermediate=15