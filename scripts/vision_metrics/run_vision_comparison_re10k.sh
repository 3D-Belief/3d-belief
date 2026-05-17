#!/usr/bin/env bash
# Run RE10K vision predictions (3D-Belief + DFoT + Gen3C) + metrics.
# Plain foreground script: run it directly, or wrap it in tmux/nohup yourself.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/tianmin-neurips/miniconda3/envs/3d-belief/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/home/ubuntu/tianmin-neurips/datasets}"
RUN_NAME="${RUN_NAME:-re10k_temporal_vision_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/vision_metrics/${RUN_NAME}}"
MODELS="${MODELS:-3d_belief,dfot,gen3c}"
STAGE="${STAGE:-test}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
NUM_EPISODES="${NUM_EPISODES:-8}"
EPISODE_FILE="${EPISODE_FILE:-}"
SCAN_MAX_FRAMES="${SCAN_MAX_FRAMES:-60}"
MAX_SCENES="${MAX_SCENES:-200}"
FVD_BACKBONE="${FVD_BACKBONE:-dfot_i3d}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BELIEF_CHECKPOINT="${BELIEF_CHECKPOINT:-/home/ubuntu/tianmin-neurips/yyin34/codebase/3d-belief/checkpoints/model-44.pt}"
BELIEF_CONFIG_PROFILE="${BELIEF_CONFIG_PROFILE:-re10k_128_vggt}"
BELIEF_OBJ_PERMANENCE_MODE="${BELIEF_OBJ_PERMANENCE_MODE:-none}"
BELIEF_OBJ_PERMANENCE_STATE_T_MIN="${BELIEF_OBJ_PERMANENCE_STATE_T_MIN:-1}"
BELIEF_OBJ_PERMANENCE_MASK_BLUR="${BELIEF_OBJ_PERMANENCE_MASK_BLUR:-5}"
BELIEF_OBJ_PERMANENCE_MASK_THRESHOLD="${BELIEF_OBJ_PERMANENCE_MASK_THRESHOLD:-0.5}"
BELIEF_OBJ_PERMANENCE_ERODE_KERNEL="${BELIEF_OBJ_PERMANENCE_ERODE_KERNEL:-0}"
BELIEF_OBJ_PERMANENCE_MASK_BINARIZE="${BELIEF_OBJ_PERMANENCE_MASK_BINARIZE:-0}"
BELIEF_DPS_GUIDANCE_SCALE="${BELIEF_DPS_GUIDANCE_SCALE:-1.0}"
BELIEF_DPS_POS_WEIGHT="${BELIEF_DPS_POS_WEIGHT:-1.0}"
BELIEF_DPS_OPACITY_WEIGHT="${BELIEF_DPS_OPACITY_WEIGHT:-0.5}"
BELIEF_REFINER_ENABLED="${BELIEF_REFINER_ENABLED:-}"
BELIEF_REFINER_NUM_ITERATIONS="${BELIEF_REFINER_NUM_ITERATIONS:-}"
BELIEF_REFINER_PRIOR_WEIGHT="${BELIEF_REFINER_PRIOR_WEIGHT:-}"
BELIEF_REFINER_DEPTH_CONSISTENCY_WEIGHT="${BELIEF_REFINER_DEPTH_CONSISTENCY_WEIGHT:-}"
BELIEF_REFINER_POSITION_UPDATE_MODE="${BELIEF_REFINER_POSITION_UPDATE_MODE:-}"
BELIEF_REFINER_RAY_TANGENT_WEIGHT="${BELIEF_REFINER_RAY_TANGENT_WEIGHT:-}"
BELIEF_REFINER_RAY_MIN_DEPTH="${BELIEF_REFINER_RAY_MIN_DEPTH:-}"
DFOT_CHECKPOINT="${DFOT_CHECKPOINT:-/home/ubuntu/tianmin-neurips/yyin34/codebase/3d-belief/checkpoints/DFoT_RE10K.ckpt}"
DFOT_REPO="${DFOT_REPO:-${REPO_ROOT}/third_party/dfot}"

GEN3C_PYTHON="${GEN3C_PYTHON:-/home/ubuntu/tianmin-neurips/miniconda3/envs/cosmos-predict1/bin/python}"
GEN3C_REPO="${GEN3C_REPO:-/home/ubuntu/tianmin-neurips/yyin34/codebase/GEN3C}"
GEN3C_CHECKPOINT_DIR="${GEN3C_CHECKPOINT_DIR:-${GEN3C_REPO}/checkpoints}"
GEN3C_CUDA_HOME="${GEN3C_CUDA_HOME:-/home/ubuntu/tianmin-neurips/miniconda3/envs/cosmos-predict1}"
GEN3C_HEIGHT="${GEN3C_HEIGHT:-704}"
GEN3C_WIDTH="${GEN3C_WIDTH:-1280}"
GEN3C_FPS="${GEN3C_FPS:-10}"
GEN3C_SEED="${GEN3C_SEED:-1}"
GEN3C_GUIDANCE="${GEN3C_GUIDANCE:-1.0}"
GEN3C_NUM_STEPS="${GEN3C_NUM_STEPS:-35}"
GEN3C_NUM_GPUS="${GEN3C_NUM_GPUS:-1}"
GEN3C_STRATEGY="${GEN3C_STRATEGY:-keyframes}"
GEN3C_FILTER_POINTS_THRESHOLD="${GEN3C_FILTER_POINTS_THRESHOLD:-0.05}"
GEN3C_FOREGROUND_MASKING="${GEN3C_FOREGROUND_MASKING:-1}"
GEN3C_ENABLE_PROMPT_ENCODER="${GEN3C_ENABLE_PROMPT_ENCODER:-0}"
GEN3C_OFFLOAD="${GEN3C_OFFLOAD:-0}"
GEN3C_MISSING_DEPTH_POLICY="${GEN3C_MISSING_DEPTH_POLICY:-moge}"

args=(
    "--dataset-root" "${DATASET_ROOT}"
    "--stage" "${STAGE}"
    "--image-size" "${IMAGE_SIZE}"
    "--models" "${MODELS}"
    "--output-root" "${OUTPUT_ROOT}"
    "--run-name" "${RUN_NAME}"
    "--num-episodes" "${NUM_EPISODES}"
    "--scan-max-frames" "${SCAN_MAX_FRAMES}"
    "--belief-checkpoint" "${BELIEF_CHECKPOINT}"
    "--belief-config-profile" "${BELIEF_CONFIG_PROFILE}"
    "--belief-obj-permanence-mode" "${BELIEF_OBJ_PERMANENCE_MODE}"
    "--belief-obj-permanence-state-t-min" "${BELIEF_OBJ_PERMANENCE_STATE_T_MIN}"
    "--belief-obj-permanence-mask-blur" "${BELIEF_OBJ_PERMANENCE_MASK_BLUR}"
    "--belief-obj-permanence-mask-threshold" "${BELIEF_OBJ_PERMANENCE_MASK_THRESHOLD}"
    "--belief-obj-permanence-erode-kernel" "${BELIEF_OBJ_PERMANENCE_ERODE_KERNEL}"
    "--belief-dps-guidance-scale" "${BELIEF_DPS_GUIDANCE_SCALE}"
    "--belief-dps-pos-weight" "${BELIEF_DPS_POS_WEIGHT}"
    "--belief-dps-opacity-weight" "${BELIEF_DPS_OPACITY_WEIGHT}"
    "--dfot-checkpoint" "${DFOT_CHECKPOINT}"
    "--dfot-repo" "${DFOT_REPO}"
    "--gen3c-python" "${GEN3C_PYTHON}"
    "--gen3c-repo" "${GEN3C_REPO}"
    "--gen3c-checkpoint-dir" "${GEN3C_CHECKPOINT_DIR}"
    "--gen3c-cuda-home" "${GEN3C_CUDA_HOME}"
    "--gen3c-height" "${GEN3C_HEIGHT}"
    "--gen3c-width" "${GEN3C_WIDTH}"
    "--gen3c-fps" "${GEN3C_FPS}"
    "--gen3c-seed" "${GEN3C_SEED}"
    "--gen3c-guidance" "${GEN3C_GUIDANCE}"
    "--gen3c-num-steps" "${GEN3C_NUM_STEPS}"
    "--gen3c-num-gpus" "${GEN3C_NUM_GPUS}"
    "--gen3c-strategy" "${GEN3C_STRATEGY}"
    "--gen3c-filter-points-threshold" "${GEN3C_FILTER_POINTS_THRESHOLD}"
    "--gen3c-missing-depth-policy" "${GEN3C_MISSING_DEPTH_POLICY}"
)
if [[ -n "${MAX_SCENES}" ]]; then
    args+=("--max-scenes" "${MAX_SCENES}")
fi
if [[ -n "${EPISODE_FILE}" ]]; then
    args+=("--episode-file" "${EPISODE_FILE}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
    args+=("--dry-run")
fi
if [[ "${BELIEF_OBJ_PERMANENCE_MASK_BINARIZE}" == "1" ]]; then
    args+=("--belief-obj-permanence-mask-binarize")
fi
if [[ "${BELIEF_REFINER_ENABLED}" == "1" ]]; then
    args+=("--belief-refiner-enabled")
elif [[ "${BELIEF_REFINER_ENABLED}" == "0" ]]; then
    args+=("--no-belief-refiner-enabled")
fi
if [[ -n "${BELIEF_REFINER_NUM_ITERATIONS}" ]]; then
    args+=("--belief-refiner-num-iterations" "${BELIEF_REFINER_NUM_ITERATIONS}")
fi
if [[ -n "${BELIEF_REFINER_PRIOR_WEIGHT}" ]]; then
    args+=("--belief-refiner-prior-weight" "${BELIEF_REFINER_PRIOR_WEIGHT}")
fi
if [[ -n "${BELIEF_REFINER_DEPTH_CONSISTENCY_WEIGHT}" ]]; then
    args+=("--belief-refiner-depth-consistency-weight" "${BELIEF_REFINER_DEPTH_CONSISTENCY_WEIGHT}")
fi
if [[ -n "${BELIEF_REFINER_POSITION_UPDATE_MODE}" ]]; then
    args+=("--belief-refiner-position-update-mode" "${BELIEF_REFINER_POSITION_UPDATE_MODE}")
fi
if [[ -n "${BELIEF_REFINER_RAY_TANGENT_WEIGHT}" ]]; then
    args+=("--belief-refiner-ray-tangent-weight" "${BELIEF_REFINER_RAY_TANGENT_WEIGHT}")
fi
if [[ -n "${BELIEF_REFINER_RAY_MIN_DEPTH}" ]]; then
    args+=("--belief-refiner-ray-min-depth" "${BELIEF_REFINER_RAY_MIN_DEPTH}")
fi
if [[ "${GEN3C_FOREGROUND_MASKING}" == "0" ]]; then
    args+=("--gen3c-no-foreground-masking")
fi
if [[ "${GEN3C_ENABLE_PROMPT_ENCODER}" == "1" ]]; then
    args+=("--gen3c-enable-prompt-encoder")
fi
if [[ "${GEN3C_OFFLOAD}" == "1" ]]; then
    args+=("--gen3c-offload")
fi

mkdir -p "${OUTPUT_ROOT}"
echo "Output: ${OUTPUT_ROOT}"

cd "${REPO_ROOT}/scripts/vision_metrics"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi

"${PYTHON_BIN}" run_vision_predictions_re10k.py "${args[@]}"
if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" compute_vision_metrics.py --run-dir "${OUTPUT_ROOT}" --fvd-backbone "${FVD_BACKBONE}"
fi
echo "Done. Metrics in ${OUTPUT_ROOT}/metrics/"
