#!/usr/bin/env bash
# Run RE10K vision predictions (3D-Belief + DFoT + Gen3C) inside a tmux session.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SESSION="${SESSION:-re10k_temporal_vision_metrics}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/re10k}"
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

BELIEF_CHECKPOINT="${BELIEF_CHECKPOINT:-${REPO_ROOT}/checkpoints/3d_belief_re10k.pt}"
DFOT_CHECKPOINT="${DFOT_CHECKPOINT:-${REPO_ROOT}/checkpoints/DFoT_RE10K.ckpt}"

GEN3C_PYTHON="${GEN3C_PYTHON:-python}"
GEN3C_REPO="${GEN3C_REPO:-${REPO_ROOT}/third_party/GEN3C}"
GEN3C_CHECKPOINT_DIR="${GEN3C_CHECKPOINT_DIR:-${GEN3C_REPO}/checkpoints}"
GEN3C_CUDA_HOME="${GEN3C_CUDA_HOME:-${CUDA_HOME:-${CONDA_PREFIX:-}}}"
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

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session '${SESSION}' already exists. Attach with: tmux attach -t ${SESSION}"
    exit 1
fi

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
    "--dfot-checkpoint" "${DFOT_CHECKPOINT}"
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
if [[ "${GEN3C_FOREGROUND_MASKING}" == "0" ]]; then
    args+=("--gen3c-no-foreground-masking")
fi
if [[ "${GEN3C_ENABLE_PROMPT_ENCODER}" == "1" ]]; then
    args+=("--gen3c-enable-prompt-encoder")
fi
if [[ "${GEN3C_OFFLOAD}" == "1" ]]; then
    args+=("--gen3c-offload")
fi

CMD_PRED="cd ${REPO_ROOT}/scripts/vision_metrics && \
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF_VALUE} \
    ${CUDA_VISIBLE_DEVICES_VALUE:+CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_VALUE}} \
    ${PYTHON_BIN} run_vision_predictions_re10k.py ${args[*]}"

CMD_METRICS="cd ${REPO_ROOT}/scripts/vision_metrics && \
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    ${PYTHON_BIN} compute_vision_metrics.py --run-dir ${OUTPUT_ROOT} --fvd-backbone ${FVD_BACKBONE}"

mkdir -p "${OUTPUT_ROOT}"
echo "Output: ${OUTPUT_ROOT}"
echo "Session: ${SESSION}"

tmux new-session -d -s "${SESSION}" -n predict "bash -lc '${CMD_PRED} && ${CMD_METRICS}; echo Done; exec bash'"
echo "tmux attach -t ${SESSION}"
