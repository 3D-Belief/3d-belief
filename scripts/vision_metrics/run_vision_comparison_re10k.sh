#!/usr/bin/env bash
# Run RE10K vision predictions + metrics for 3D-Belief, DFoT, and Gen3C.
# Plain foreground script: run it directly, or wrap it in tmux/nohup yourself.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-3d-belief}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/re10k}"
RUN_NAME="${RUN_NAME:-re10k_temporal_vision_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/vision_metrics/${RUN_NAME}}"
MODELS="${MODELS:-3d_belief,dfot}"
STAGE="${STAGE:-test}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
NUM_EPISODES="${NUM_EPISODES:-200}"
SCAN_MAX_FRAMES="${SCAN_MAX_FRAMES:-60}"
MAX_SCENES="${MAX_SCENES:-200}"
FVD_BACKBONE="${FVD_BACKBONE:-dfot_i3d}"
DRY_RUN="${DRY_RUN:-0}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BELIEF_CHECKPOINT="${BELIEF_CHECKPOINT:-${REPO_ROOT}/checkpoints/3d_belief_re10k_256.pt}"
BELIEF_CONFIG_PROFILE="${BELIEF_CONFIG_PROFILE:-re10k_256_from_128}"
DFOT_CHECKPOINT="${DFOT_CHECKPOINT:-${REPO_ROOT}/checkpoints/DFoT_RE10K.ckpt}"

# Gen3C settings (only used when gen3c is included in MODELS).
GEN3C_REPO="${GEN3C_REPO:-${REPO_ROOT}/third_party/gen3c}"
GEN3C_PYTHON="${GEN3C_PYTHON:-python}"
GEN3C_CHECKPOINT_DIR="${GEN3C_CHECKPOINT_DIR:-${GEN3C_REPO}/checkpoints}"
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
    "--dfot-checkpoint" "${DFOT_CHECKPOINT}"
    "--gen3c-python" "${GEN3C_PYTHON}"
    "--gen3c-repo" "${GEN3C_REPO}"
    "--gen3c-checkpoint-dir" "${GEN3C_CHECKPOINT_DIR}"
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
if [[ -n "${MAX_SCENES:-}" ]]; then
    args+=("--max-scenes" "${MAX_SCENES}")
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

# Resolve the interpreter: explicit PYTHON_BIN, else the conda env, else venv.
if [[ -x "${PYTHON_BIN}" ]]; then
    VISION_METRICS_PYTHON="${PYTHON_BIN}"
elif command -v conda >/dev/null 2>&1; then
    set +u  # conda's activate.d scripts reference unbound vars
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    set -u
    VISION_METRICS_PYTHON="${CONDA_PREFIX}/bin/python"
elif [[ -f "${REPO_ROOT}/uv_venv/.venv/bin/activate" ]]; then
    source "${REPO_ROOT}/uv_venv/.venv/bin/activate"
    VISION_METRICS_PYTHON="python"
else
    VISION_METRICS_PYTHON="python"
fi

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/splat_belief:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="/tmp/matplotlib"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"

mkdir -p "${OUTPUT_ROOT}"
echo "Run directory:        ${OUTPUT_ROOT}"
echo "Models:               ${MODELS}"
echo "Checkpoint:           ${BELIEF_CHECKPOINT}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"${VISION_METRICS_PYTHON}" -u scripts/vision_metrics/run_vision_predictions_re10k.py "${args[@]}"
if [[ "${DRY_RUN}" != "1" ]]; then
    "${VISION_METRICS_PYTHON}" -u scripts/vision_metrics/compute_vision_metrics.py \
        --run-dir "${OUTPUT_ROOT}" --fvd-backbone "${FVD_BACKBONE}"
fi

echo "Done. Metrics in ${OUTPUT_ROOT}/metrics/"
