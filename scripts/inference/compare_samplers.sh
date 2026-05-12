#!/bin/bash
# A/B compare diffusion samplers on the scenes listed in
# splat_belief/config/inference/temporal_indices.py.
#
# Default comparison (override with CKPT, GPU, RES_FOLDER env vars):
#   - ddim          @ 50 steps   (legacy)
#   - dpm_solver_pp @ 15 steps   (~3-5x faster, similar quality)
#
# Outputs:
#   ${RES_FOLDER}/{ddim_50,dpm_solver_pp_15}/visuals_*/...
#   ${RES_FOLDER}/comparison_summary.json
#   ${RES_FOLDER}/comparison_summary.txt
# Per-scene `timing.json` is written inside each visuals_* folder.

eval "$(conda shell.bash hook)"
conda activate 3d-belief

nvidia-smi | head -3

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TIPS_REPO="${TIPS_REPO:-${REPO_ROOT}}"

CKPT="${CKPT:-${REPO_ROOT}/outputs/weights/semantic/model-15.pt}"
GPU="${GPU:-3}"
RES_FOLDER="${RES_FOLDER:-${REPO_ROOT}/outputs/inference/sampler_compare}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/all_rerendered_root}"

export MASTER_PORT=$((12000 + RANDOM % 1000))
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=${GPU}
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export PYTHONPATH="${REPO_ROOT}:${TIPS_REPO}:${PYTHONPATH}"
export WANDB_MODE=disabled

cd "${REPO_ROOT}"

echo "checkpoint:      ${CKPT}"
echo "gpu:             ${GPU}"
echo "results folder:  ${RES_FOLDER}"

python splat_belief/experiment/compare_samplers.py \
    dataset=spoc_seq \
    dataset.root_dir="${DATASET_ROOT}" \
    max_scenes=5 \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    stage=train \
    model/encoder=uvitmvsplat \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=true \
    model.encoder.use_reg_model=true \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=768 \
    model.encoder.reg_model_name=dinov3_base \
    model.encoder.reg_model_weights=${REPO_ROOT}/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.inference_mode=false \
    model.encoder.use_depth_mask=false \
    model.encoder.freeze_depth_predictor=false \
    model.encoder.grid_sample_disable_cudnn=true \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.use_vggt_alignment=false \
    model.encoder.backbone.use_repa=false \
    model.encoder.backbone.input_size='[128, 128]' \
    repa_encoder_name=dinov3-vit-b \
    repa_encoder_weights=${REPO_ROOT}/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    semantic_mode=embed \
    semantic_viz=query \
    temperature=0.85 \
    sampling_steps=50 \
    name=sampler_compare \
    image_size=128 \
    inference_sample_from_dataset=false \
    inference_num_samples=25 \
    inference_min_frames=5 \
    inference_max_frames=25 \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    clean_target=false \
    use_history=false \
    inference_save_scene=false \
    semantic_config=${REPO_ROOT}/splat_belief/config/semantic/onehot.yaml \
    checkpoint_path=${CKPT} \
    results_folder=${RES_FOLDER} \
    wandb=local \
    "+compare_samplers=[{name: ddim, steps: 50, dtype: fp32}, {name: ddim, steps: 50, dtype: bf16}, {name: dpm_solver_pp, steps: 15, dtype: fp32}, {name: dpm_solver_pp, steps: 15, dtype: bf16}]"
