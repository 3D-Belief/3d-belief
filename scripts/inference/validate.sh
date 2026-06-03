#!/usr/bin/env bash
# Run 3D-Belief temporal inference / validation on the SPOC sequence dataset.
# Only the paths below are meant to be overridden (via env vars); everything
# else is a fixed, known-good default.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

# --- Overridable paths ---
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/spoc}"
STAGE="${STAGE:-test}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/3d_belief_spoc.pt}"
RESULTS_FOLDER="${RESULTS_FOLDER:-outputs/inference/spoc_base}"

# --- Overridable model resolution (128 default; set IMAGE_SIZE=256 for 256 ckpts) ---
IMAGE_SIZE="${IMAGE_SIZE:-128}"

# --- Overridable sampling controls ---
INFERENCE_SAMPLE_FROM_DATASET="${INFERENCE_SAMPLE_FROM_DATASET:-true}"
INFERENCE_NUM_SAMPLES="${INFERENCE_NUM_SAMPLES:-25}"
INFERENCE_SAVE_SCENE="${INFERENCE_SAVE_SCENE:-false}"

# --- Overridable object-permanence guidance (none | opacity | dps) ---
OBJ_PERMANENCE_MODE="${OBJ_PERMANENCE_MODE:-none}"
# --- Overridable observed-side object permanence (none | live) ---
OBJ_PERMANENCE_OBSERVED_MODE="${OBJ_PERMANENCE_OBSERVED_MODE:-live}"

# --- Environment ---
if [[ -f uv_venv/.venv/bin/activate ]]; then
    set +u; source uv_venv/.venv/bin/activate; set -u
elif command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV:-3d-belief}"
    set -u
else
    echo "ERROR: activate uv_venv/.venv or install conda before running validation." >&2
    exit 1
fi

nvidia-smi

ENV_PREFIX="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
export MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 1000))}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-${ENV_PREFIX}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;9.0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS_FOLDER}"

python splat_belief/experiment/temporal_inference.py \
    dataset=spoc_seq \
    dataset.root_dir="${DATASET_ROOT}" \
    stage="${STAGE}" \
    checkpoint_path="${CHECKPOINT_PATH}" \
    results_folder="${RESULTS_FOLDER}" \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
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
    model.encoder.reg_model_weights="${REPO_ROOT}/checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth" \
    model.encoder.use_depth_mask=false \
    model.encoder.grid_sample_disable_cudnn=true \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.inference_mode=false \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.input_size="[${IMAGE_SIZE}, ${IMAGE_SIZE}]" \
    model_type=uvit_pose \
    semantic_mode=embed \
    semantic_viz=query \
    semantic_config=splat_belief/config/semantic/onehot.yaml \
    temperature=0.85 \
    sampler=dpm_solver_pp \
    sampling_steps=15 \
    inference_dtype=fp32 \
    image_size=${IMAGE_SIZE} \
    name=mvsplat_inference \
    inference_sample_from_dataset="${INFERENCE_SAMPLE_FROM_DATASET}" \
    inference_num_samples="${INFERENCE_NUM_SAMPLES}" \
    obj_permanence_mode="${OBJ_PERMANENCE_MODE}" \
    obj_permanence_observed_mode="${OBJ_PERMANENCE_OBSERVED_MODE}" \
    inference_min_frames=5 \
    inference_max_frames=25 \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    clean_target=false \
    use_history=false \
    inference_save_scene="${INFERENCE_SAVE_SCENE}"
