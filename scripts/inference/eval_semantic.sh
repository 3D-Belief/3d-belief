#!/bin/bash
# Evaluate the latest semantic-finetune checkpoint:
#   - per-pixel CLIP query similarity heatmaps (open-vocab semantic head)
#   - PCA-3 of rendered DINOv3 reg features vs GT DINOv3 features
# Outputs panels to ${REPO_ROOT}/outputs/training/spoc_semantic/eval/

source uv_venv/.venv/bin/activate

nvidia-smi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/all_rerendered_root}"

export MASTER_PORT=$((12000 + RANDOM % 1000))
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

# Default: pick the highest-numbered model-*.pt produced by finetune_semantic.sh.
# Override with CKPT=<model.pt> if you want a specific one.
if [[ -z "${CKPT:-}" ]]; then
    CKPT="$(ls ${REPO_ROOT}/outputs/training/spoc_semantic/model-*.pt 2>/dev/null \
            | sed 's/.*model-\([0-9]*\)\.pt/\1 &/' | sort -n | tail -1 | cut -d' ' -f2-)"
fi
if [[ -z "${CKPT}" ]]; then
    echo "ERROR: no model-*.pt found in ${REPO_ROOT}/outputs/training/spoc_semantic/"
    exit 1
fi
echo "Using checkpoint: ${CKPT}"

python splat_belief/experiment/eval_semantic.py \
    dataset=spoc \
    dataset.vggt_alignment_loss_weight=2.0 \
    dataset.intermediate_weight=5.0 \
    dataset.depth_smooth_loss_weight=0.1 \
    dataset.depth_loss_weight=1.0 \
    dataset.root_dir="${DATASET_ROOT}" \
    setting_name=pixelsplat_h100 \
    stage=train \
    results_folder=${REPO_ROOT}/outputs/training/spoc_semantic \
    semantic_config=${REPO_ROOT}/splat_belief/config/semantic/onehot.yaml \
    checkpoint_path=${CKPT} \
    ngpus=1 \
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
    model.encoder.grid_sample_disable_cudnn=true \
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
    name=spoc_semantic_eval \
    wandb=local \
    clean_target=false \
    use_identity=true \
    intermediate=true \
    load_optimizer=false \
    load_enc=false \
    use_depth_smoothness=true \
    adjacent_angle=0.785 \
    adjacent_distance=1.0 \
    num_intermediate=15 \
    semantic_mode=embed \
    +eval_num_samples=8 \
    "+eval_query_labels=[bed, sofa, chair, table, tv, kitchen counter, wall, floor]"
