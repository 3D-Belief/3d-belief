#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SESSION="${SESSION:-spoc_temporal_vision_metrics}"
CONDA_ENV="${CONDA_ENV:-3d-belief}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/data/spoc}"
RUN_NAME="${RUN_NAME:-spoc_temporal_vision_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/vision_metrics/${RUN_NAME}}"
MODELS="${MODELS:-3d_belief,nwm,dfot}"
STAGE="${STAGE:-test}"
NUM_EPISODES="${NUM_EPISODES:-8}"
SCAN_MAX_FRAMES="${SCAN_MAX_FRAMES:-80}"
FVD_BACKBONE="${FVD_BACKBONE:-dfot_i3d}"
DRY_RUN="${DRY_RUN:-0}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES:-}"
PYTORCH_CUDA_ALLOC_CONF_VALUE="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BELIEF_CHECKPOINT="${BELIEF_CHECKPOINT:-${REPO_ROOT}/outputs/weights/semantic/model-15.pt}"
DFOT_CHECKPOINT="${DFOT_CHECKPOINT:-${REPO_ROOT}/checkpoints/dfot_finetune_spoc.ckpt}"
NWM_CHECKPOINT="${NWM_CHECKPOINT:-${REPO_ROOT}/checkpoints/nwm_finetune_spoc.pth.tar}"
NWM_OBSERVED_MODE="${NWM_OBSERVED_MODE:-diffusion_guidance}"
NWM_IMAGINED_MODE="${NWM_IMAGINED_MODE:-multi_horizon}"
GEN3C_REPO="${GEN3C_REPO:-${REPO_ROOT}/third_party/GEN3C}"
GEN3C_PYTHON="${GEN3C_PYTHON:-python}"
GEN3C_CHECKPOINT_DIR="${GEN3C_CHECKPOINT_DIR:-${GEN3C_REPO}/checkpoints}"
GEN3C_CUDA_HOME="${GEN3C_CUDA_HOME:-${CUDA_HOME:-${CONDA_PREFIX:-}}}"
GEN3C_HEIGHT="${GEN3C_HEIGHT:-704}"
GEN3C_WIDTH="${GEN3C_WIDTH:-1280}"
GEN3C_FPS="${GEN3C_FPS:-10}"
GEN3C_SEED="${GEN3C_SEED:-1}"
GEN3C_GUIDANCE="${GEN3C_GUIDANCE:-1}"
GEN3C_NUM_STEPS="${GEN3C_NUM_STEPS:-35}"
GEN3C_NUM_GPUS="${GEN3C_NUM_GPUS:-1}"
GEN3C_STRATEGY="${GEN3C_STRATEGY:-keyframes}"
GEN3C_FILTER_POINTS_THRESHOLD="${GEN3C_FILTER_POINTS_THRESHOLD:-0.05}"
GEN3C_FOREGROUND_MASKING="${GEN3C_FOREGROUND_MASKING:-1}"
GEN3C_OFFLOAD="${GEN3C_OFFLOAD:-0}"
GEN3C_ENABLE_PROMPT_ENCODER="${GEN3C_ENABLE_PROMPT_ENCODER:-0}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session '${SESSION}' already exists. Attach with: tmux attach -t ${SESSION}"
    exit 1
fi

args=(
    "--dataset-root" "${DATASET_ROOT}"
    "--stage" "${STAGE}"
    "--models" "${MODELS}"
    "--output-root" "${OUTPUT_ROOT}"
    "--run-name" "${RUN_NAME}"
    "--num-episodes" "${NUM_EPISODES}"
    "--scan-max-frames" "${SCAN_MAX_FRAMES}"
    "--belief-checkpoint" "${BELIEF_CHECKPOINT}"
    "--dfot-checkpoint" "${DFOT_CHECKPOINT}"
    "--nwm-checkpoint" "${NWM_CHECKPOINT}"
    "--nwm-observed-mode" "${NWM_OBSERVED_MODE}"
    "--nwm-imagined-mode" "${NWM_IMAGINED_MODE}"
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
)

if [[ "${DRY_RUN}" == "1" ]]; then
    args+=("--dry-run")
fi
if [[ "${GEN3C_FOREGROUND_MASKING}" != "1" ]]; then
    args+=("--gen3c-no-foreground-masking")
fi
if [[ "${GEN3C_ENABLE_PROMPT_ENCODER}" == "1" ]]; then
    args+=("--gen3c-enable-prompt-encoder")
fi
if [[ "${GEN3C_OFFLOAD}" == "1" ]]; then
    args+=("--gen3c-offload")
fi

mkdir -p "${OUTPUT_ROOT}"
command_file="${OUTPUT_ROOT}/tmux_eval_command.sh"

cat > "${command_file}" <<EOF
#!/usr/bin/env bash
set -eo pipefail
cd "${REPO_ROOT}"
if [[ -x "${PYTHON_BIN}" ]]; then
    export VISION_METRICS_PYTHON="${PYTHON_BIN}"
elif command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook)"
    conda activate "${CONDA_ENV}"
    export VISION_METRICS_PYTHON="python"
elif [[ -f "uv_venv/.venv/bin/activate" ]]; then
    source "uv_venv/.venv/bin/activate"
    export VISION_METRICS_PYTHON="python"
else
    export VISION_METRICS_PYTHON="python"
fi
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/splat_belief:\${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Avoid user-site torch ABI conflicts.
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="/tmp/matplotlib"
export OPENBLAS_NUM_THREADS="\${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="\${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="\${MKL_NUM_THREADS:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF_VALUE}"
if [[ -n "${CUDA_VISIBLE_DEVICES_VALUE}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_VALUE}"
fi
echo "Run directory: ${OUTPUT_ROOT}"
echo "Models: ${MODELS}"
echo "CUDA_VISIBLE_DEVICES: \${CUDA_VISIBLE_DEVICES:-unset}"
"\${VISION_METRICS_PYTHON}" -u scripts/vision_metrics/run_vision_predictions.py ${args[*]}
if [[ "${DRY_RUN}" != "1" ]]; then
    "\${VISION_METRICS_PYTHON}" -u scripts/vision_metrics/compute_vision_metrics.py --run-dir "${OUTPUT_ROOT}" --fvd-backbone "${FVD_BACKBONE}"
fi
EOF
chmod +x "${command_file}"

tmux new-session -d -s "${SESSION}" "bash -lc 'set +e; \"${command_file}\"; status=\$?; echo; echo temporal vision metrics command exited with status \$status; exec bash'"

echo "Started tmux session: ${SESSION}"
echo "Attach with: tmux attach -t ${SESSION}"
echo "Run directory: ${OUTPUT_ROOT}"
