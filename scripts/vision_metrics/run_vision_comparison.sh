#!/usr/bin/env bash
# Run SPOC (AI2-THOR) vision predictions + metrics for 3D-Belief, NWM, and DFoT.
# Plain foreground script: run it directly, or wrap it in tmux/nohup yourself.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-3d-belief}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/spoc}"
RUN_NAME="${RUN_NAME:-spoc_temporal_vision_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/vision_metrics/${RUN_NAME}}"
MODELS="${MODELS:-3d_belief,nwm,dfot}"
STAGE="${STAGE:-test}"
NUM_EPISODES="${NUM_EPISODES:-200}"
SCAN_MAX_FRAMES="${SCAN_MAX_FRAMES:-80}"
FVD_BACKBONE="${FVD_BACKBONE:-dfot_i3d}"
DRY_RUN="${DRY_RUN:-0}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BELIEF_CHECKPOINT="${BELIEF_CHECKPOINT:-${REPO_ROOT}/checkpoints/3d_belief_spoc.pt}"
BELIEF_OBJ_PERMANENCE_MODE="${BELIEF_OBJ_PERMANENCE_MODE:-none}"
BELIEF_OBJ_PERMANENCE_OBSERVED_MODE="${BELIEF_OBJ_PERMANENCE_OBSERVED_MODE:-live}"
DFOT_CHECKPOINT="${DFOT_CHECKPOINT:-${REPO_ROOT}/checkpoints/dfot_finetune_spoc.ckpt}"
NWM_CHECKPOINT="${NWM_CHECKPOINT:-${REPO_ROOT}/checkpoints/nwm_finetune_spoc.pth.tar}"
NWM_OBSERVED_MODE="${NWM_OBSERVED_MODE:-diffusion_guidance}"
NWM_IMAGINED_MODE="${NWM_IMAGINED_MODE:-multi_horizon}"

args=(
    "--dataset-root" "${DATASET_ROOT}"
    "--stage" "${STAGE}"
    "--models" "${MODELS}"
    "--output-root" "${OUTPUT_ROOT}"
    "--run-name" "${RUN_NAME}"
    "--num-episodes" "${NUM_EPISODES}"
    "--scan-max-frames" "${SCAN_MAX_FRAMES}"
    "--belief-checkpoint" "${BELIEF_CHECKPOINT}"
    "--belief-obj-permanence-mode" "${BELIEF_OBJ_PERMANENCE_MODE}"
    "--belief-obj-permanence-observed-mode" "${BELIEF_OBJ_PERMANENCE_OBSERVED_MODE}"
    "--dfot-checkpoint" "${DFOT_CHECKPOINT}"
    "--nwm-checkpoint" "${NWM_CHECKPOINT}"
    "--nwm-observed-mode" "${NWM_OBSERVED_MODE}"
    "--nwm-imagined-mode" "${NWM_IMAGINED_MODE}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
    args+=("--dry-run")
fi

# Resolve the interpreter: explicit PYTHON_BIN, else the conda env, else venv.
if [[ -x "${PYTHON_BIN}" ]]; then
    VISION_METRICS_PYTHON="${PYTHON_BIN}"
elif command -v conda >/dev/null 2>&1; then
    set +u  # conda's activate.d scripts reference unbound vars
    eval "$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    set -u
    # conda activate may not reorder PATH in non-interactive shells; use the
    # env interpreter explicitly.
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
echo "3D-Belief obj perm:   ${BELIEF_OBJ_PERMANENCE_MODE} (${BELIEF_OBJ_PERMANENCE_OBSERVED_MODE})"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

"${VISION_METRICS_PYTHON}" -u scripts/vision_metrics/run_vision_predictions.py "${args[@]}"
if [[ "${DRY_RUN}" != "1" ]]; then
    "${VISION_METRICS_PYTHON}" -u scripts/vision_metrics/compute_vision_metrics.py \
        --run-dir "${OUTPUT_ROOT}" --fvd-backbone "${FVD_BACKBONE}"
fi

echo "Done. Metrics in ${OUTPUT_ROOT}/metrics/"
