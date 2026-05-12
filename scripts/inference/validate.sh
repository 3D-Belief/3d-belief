#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

if [[ -f uv_venv/.venv/bin/activate ]]; then
    # Legacy venv support.
    set +u
    source uv_venv/.venv/bin/activate
    set -u
    VALIDATE_ENV_KIND="venv"
elif command -v conda >/dev/null 2>&1; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "${VALIDATE_CONDA_ENV:-3d-belief}"
    set -u
    VALIDATE_ENV_KIND="conda"
else
    echo "ERROR: activate uv_venv/.venv or install conda before running validation." >&2
    exit 1
fi

nvidia-smi

export MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 1000))}"
ENV_PREFIX="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
export PATH="${ENV_PREFIX}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME:-${ENV_PREFIX}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;9.0}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

DATASET="${DATASET:-spoc_seq}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/all_rerendered_root}"
STAGE="${STAGE:-test}"
SETTING_NAME="${SETTING_NAME:-}"
RESULTS_FOLDER="${RESULTS_FOLDER:-outputs/inference/spoc_base}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${REPO_ROOT}/checkpoints/3d_belief_spoc.pt}"

SAMPLER="${SAMPLER:-ddim}"
SAMPLING_STEPS="${SAMPLING_STEPS:-50}"
INFERENCE_DTYPE="${INFERENCE_DTYPE:-fp32}"
TEMPERATURE="${TEMPERATURE:-0.85}"

IMAGE_SIZE="${IMAGE_SIZE:-128}"
INPUT_SIZE="${INPUT_SIZE:-[128, 128]}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_TARGET="${NUM_TARGET:-1}"
NUM_CONTEXT="${NUM_CONTEXT:-1}"
CTXT_MIN="${CTXT_MIN:-}"
CTXT_MAX="${CTXT_MAX:-}"

USE_IMAGE_CONDITION="${USE_IMAGE_CONDITION:-true}"
DEPTH_PREDICTOR_TIME_EMBED="${DEPTH_PREDICTOR_TIME_EMBED:-true}"
EVOLVE_CTXT="${EVOLVE_CTXT:-false}"
USE_CAMERA_POSE="${USE_CAMERA_POSE:-true}"
USE_SEMANTIC="${USE_SEMANTIC:-true}"
USE_REG_MODEL="${USE_REG_MODEL:-true}"
USE_DEPTH_MASK="${USE_DEPTH_MASK:-}"
FREEZE_DEPTH_PREDICTOR="${FREEZE_DEPTH_PREDICTOR:-}"
GRID_SAMPLE_DISABLE_CUDNN="${GRID_SAMPLE_DISABLE_CUDNN:-}"
USE_VGGT_ALIGNMENT="${USE_VGGT_ALIGNMENT:-}"
USE_REPA="${USE_REPA:-}"
D_SEMANTIC="${D_SEMANTIC:-512}"
D_SEMANTIC_REG="${D_SEMANTIC_REG:-384}"
GAUSSIANS_PER_PIXEL="${GAUSSIANS_PER_PIXEL:-1}"
INFERENCE_MODE="${INFERENCE_MODE:-false}"
MODEL_TYPE="${MODEL_TYPE:-uvit_pose}"
SEMANTIC_MODE="${SEMANTIC_MODE:-embed}"
SEMANTIC_VIZ="${SEMANTIC_VIZ:-query}"
SEMANTIC_CONFIG="${SEMANTIC_CONFIG:-splat_belief/config/semantic/onehot.yaml}"
NAME="${NAME:-mvsplat_inference}"

INFERENCE_SAMPLE_FROM_DATASET="${INFERENCE_SAMPLE_FROM_DATASET:-true}"
INFERENCE_INDICES="${INFERENCE_INDICES:-}"
INFERENCE_NUM_SAMPLES="${INFERENCE_NUM_SAMPLES:-25}"
INFERENCE_MIN_FRAMES="${INFERENCE_MIN_FRAMES:-5}"
INFERENCE_MAX_FRAMES="${INFERENCE_MAX_FRAMES:-25}"
INFERENCE_SAVE_SCENE="${INFERENCE_SAVE_SCENE:-true}"

ADJACENT_ANGLE="${ADJACENT_ANGLE:-0.523}"
ADJACENT_DISTANCE="${ADJACENT_DISTANCE:-1.0}"
CLEAN_TARGET="${CLEAN_TARGET:-false}"
USE_HISTORY="${USE_HISTORY:-false}"
USE_DEPTH_SUPERVISION="${USE_DEPTH_SUPERVISION:-}"
USE_IDENTITY="${USE_IDENTITY:-}"
INTERMEDIATE="${INTERMEDIATE:-}"
NUM_INTERMEDIATE="${NUM_INTERMEDIATE:-}"
CTXT_LOSSES_FACTOR="${CTXT_LOSSES_FACTOR:-}"
REPA_ENCODER_RESOLUTION="${REPA_ENCODER_RESOLUTION:-}"
ALIGNMENT_LATENTS_INFO="${ALIGNMENT_LATENTS_INFO:-}"
DATASET_VGGT_ALIGNMENT_LOSS_WEIGHT="${DATASET_VGGT_ALIGNMENT_LOSS_WEIGHT:-}"
DATASET_INTERMEDIATE_WEIGHT="${DATASET_INTERMEDIATE_WEIGHT:-}"
DATASET_DEPTH_SMOOTH_LOSS_WEIGHT="${DATASET_DEPTH_SMOOTH_LOSS_WEIGHT:-}"
DATASET_DEPTH_LOSS_WEIGHT="${DATASET_DEPTH_LOSS_WEIGHT:-}"
OBJ_PERMANENCE_MODE="${OBJ_PERMANENCE_MODE:-}"
OBJ_PERMANENCE_STATE_T_MIN="${OBJ_PERMANENCE_STATE_T_MIN:-}"
OBJ_PERMANENCE_MASK_BLUR="${OBJ_PERMANENCE_MASK_BLUR:-}"
OBJ_PERMANENCE_MASK_THRESHOLD="${OBJ_PERMANENCE_MASK_THRESHOLD:-}"
OBJ_PERMANENCE_ERODE_KERNEL="${OBJ_PERMANENCE_ERODE_KERNEL:-}"
DPS_GUIDANCE_SCALE="${DPS_GUIDANCE_SCALE:-}"
DPS_POS_WEIGHT="${DPS_POS_WEIGHT:-}"
DPS_OPACITY_WEIGHT="${DPS_OPACITY_WEIGHT:-}"

mkdir -p "${RESULTS_FOLDER}"

args=(
    "dataset=${DATASET}"
    "dataset.root_dir=${DATASET_ROOT}"
    "batch_size=${BATCH_SIZE}"
    "num_target=${NUM_TARGET}"
    "num_context=${NUM_CONTEXT}"
    "stage=${STAGE}"
    "model/encoder=uvitmvsplat"
    "model.encoder.use_image_condition=${USE_IMAGE_CONDITION}"
    "model.encoder.depth_predictor_time_embed=${DEPTH_PREDICTOR_TIME_EMBED}"
    "model.encoder.evolve_ctxt=${EVOLVE_CTXT}"
    "model.encoder.use_camera_pose=${USE_CAMERA_POSE}"
    "model.encoder.use_semantic=${USE_SEMANTIC}"
    "model.encoder.use_reg_model=${USE_REG_MODEL}"
    "model.encoder.d_semantic=${D_SEMANTIC}"
    "model.encoder.d_semantic_reg=${D_SEMANTIC_REG}"
    "model.encoder.gaussians_per_pixel=${GAUSSIANS_PER_PIXEL}"
    "model.encoder.inference_mode=${INFERENCE_MODE}"
    "model/encoder/backbone=u_vit3d_pose"
    "model.encoder.backbone.input_size=${INPUT_SIZE}"
    "model_type=${MODEL_TYPE}"
    "semantic_mode=${SEMANTIC_MODE}"
    "semantic_viz=${SEMANTIC_VIZ}"
    "temperature=${TEMPERATURE}"
    "sampling_steps=${SAMPLING_STEPS}"
    "sampler=${SAMPLER}"
    "inference_dtype=${INFERENCE_DTYPE}"
    "name=${NAME}"
    "image_size=${IMAGE_SIZE}"
    "inference_sample_from_dataset=${INFERENCE_SAMPLE_FROM_DATASET}"
    "inference_num_samples=${INFERENCE_NUM_SAMPLES}"
    "inference_min_frames=${INFERENCE_MIN_FRAMES}"
    "inference_max_frames=${INFERENCE_MAX_FRAMES}"
    "adjacent_angle=${ADJACENT_ANGLE}"
    "adjacent_distance=${ADJACENT_DISTANCE}"
    "clean_target=${CLEAN_TARGET}"
    "use_history=${USE_HISTORY}"
    "inference_save_scene=${INFERENCE_SAVE_SCENE}"
    "semantic_config=${SEMANTIC_CONFIG}"
    "checkpoint_path=${CHECKPOINT_PATH}"
    "results_folder=${RESULTS_FOLDER}"
)

[[ -n "${CTXT_MIN}" ]] && args+=("ctxt_min=${CTXT_MIN}")
[[ -n "${CTXT_MAX}" ]] && args+=("ctxt_max=${CTXT_MAX}")
[[ -n "${SETTING_NAME}" ]] && args+=("setting_name=${SETTING_NAME}")
[[ -n "${INFERENCE_INDICES}" ]] && args+=("inference_indices=${INFERENCE_INDICES}")
[[ -n "${USE_DEPTH_MASK}" ]] && args+=("model.encoder.use_depth_mask=${USE_DEPTH_MASK}")
[[ -n "${FREEZE_DEPTH_PREDICTOR}" ]] && args+=("model.encoder.freeze_depth_predictor=${FREEZE_DEPTH_PREDICTOR}")
[[ -n "${GRID_SAMPLE_DISABLE_CUDNN}" ]] && args+=("model.encoder.grid_sample_disable_cudnn=${GRID_SAMPLE_DISABLE_CUDNN}")
[[ -n "${USE_VGGT_ALIGNMENT}" ]] && args+=("model.encoder.backbone.use_vggt_alignment=${USE_VGGT_ALIGNMENT}")
[[ -n "${USE_REPA}" ]] && args+=("model.encoder.backbone.use_repa=${USE_REPA}")
[[ -n "${USE_DEPTH_SUPERVISION}" ]] && args+=("use_depth_supervision=${USE_DEPTH_SUPERVISION}")
[[ -n "${USE_IDENTITY}" ]] && args+=("use_identity=${USE_IDENTITY}")
[[ -n "${INTERMEDIATE}" ]] && args+=("intermediate=${INTERMEDIATE}")
[[ -n "${NUM_INTERMEDIATE}" ]] && args+=("num_intermediate=${NUM_INTERMEDIATE}")
[[ -n "${CTXT_LOSSES_FACTOR}" ]] && args+=("ctxt_losses_factor=${CTXT_LOSSES_FACTOR}")
[[ -n "${REPA_ENCODER_RESOLUTION}" ]] && args+=("repa_encoder_resolution=${REPA_ENCODER_RESOLUTION}")
[[ -n "${ALIGNMENT_LATENTS_INFO}" ]] && args+=("alignment.latents_info=${ALIGNMENT_LATENTS_INFO}")
[[ -n "${DATASET_VGGT_ALIGNMENT_LOSS_WEIGHT}" ]] && args+=("dataset.vggt_alignment_loss_weight=${DATASET_VGGT_ALIGNMENT_LOSS_WEIGHT}")
[[ -n "${DATASET_INTERMEDIATE_WEIGHT}" ]] && args+=("dataset.intermediate_weight=${DATASET_INTERMEDIATE_WEIGHT}")
[[ -n "${DATASET_DEPTH_SMOOTH_LOSS_WEIGHT}" ]] && args+=("dataset.depth_smooth_loss_weight=${DATASET_DEPTH_SMOOTH_LOSS_WEIGHT}")
[[ -n "${DATASET_DEPTH_LOSS_WEIGHT}" ]] && args+=("dataset.depth_loss_weight=${DATASET_DEPTH_LOSS_WEIGHT}")
[[ -n "${OBJ_PERMANENCE_MODE}" ]] && args+=("obj_permanence_mode=${OBJ_PERMANENCE_MODE}")
[[ -n "${OBJ_PERMANENCE_STATE_T_MIN}" ]] && args+=("obj_permanence_state_t_min=${OBJ_PERMANENCE_STATE_T_MIN}")
[[ -n "${OBJ_PERMANENCE_MASK_BLUR}" ]] && args+=("obj_permanence_mask_blur=${OBJ_PERMANENCE_MASK_BLUR}")
[[ -n "${OBJ_PERMANENCE_MASK_THRESHOLD}" ]] && args+=("obj_permanence_mask_threshold=${OBJ_PERMANENCE_MASK_THRESHOLD}")
[[ -n "${OBJ_PERMANENCE_ERODE_KERNEL}" ]] && args+=("obj_permanence_erode_kernel=${OBJ_PERMANENCE_ERODE_KERNEL}")
[[ -n "${DPS_GUIDANCE_SCALE}" ]] && args+=("dps_guidance_scale=${DPS_GUIDANCE_SCALE}")
[[ -n "${DPS_POS_WEIGHT}" ]] && args+=("dps_pos_weight=${DPS_POS_WEIGHT}")
[[ -n "${DPS_OPACITY_WEIGHT}" ]] && args+=("dps_opacity_weight=${DPS_OPACITY_WEIGHT}")

printf 'Validation args:\n'
printf '  %q\n' "${args[@]}"

python splat_belief/experiment/temporal_inference.py "${args[@]}"

if [[ "${VALIDATE_ENV_KIND}" == "venv" ]]; then
    set +u
    deactivate || true
    set -u
else
    set +u
    conda deactivate || true
    set -u
fi
