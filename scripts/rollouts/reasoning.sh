#!/bin/bash
#SBATCH --job-name=reasoning
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=125G
#SBATCH --time=24:00:00
#SBATCH --output=wm_baselines/output/_logs/reasoning_%j.out
#SBATCH --error=wm_baselines/output/_logs/reasoning_%j.err

set -euo pipefail

# A small wrapper to run the different reasoning workspaces by agent key.
# Usage: bash reasoning.sh <AGENT> [EXTRA_HYDRA_ARGS...]

# Root of this repository (two levels up from scripts/rollouts)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/wm_baselines/workspace"
OUTPUT_DIR="${REPO_ROOT}/outputs"
CONDA_ENV="3d-belief"

# Environment variables (kept from original script)
export XFORMERS_DISABLED=1
export CUDA_VISIBLE_DEVICES=0
export OBJAVERSE_DATA_DIR="data/2023_07_28"
export OBJAVERSE_HOUSES_DIR="data/houses_2023_07_28"

get_agent_config() {
    local agent="$1"
    # Sets SCRIPT_FILE, SAVE_NAME and optional EPISODE_OVERRIDE
    case "${agent}" in
        obj_comp_3d_belief)
            SCRIPT_FILE="obj_comp/spoc_obj_completion_3d_belief_workspace.py"
            SAVE_NAME="spoc_obj_completion_3d_belief"
            EPISODE_OVERRIDE="${REPO_ROOT}/data/3d-core/object_completion"
            ;;
        room_comp_3d_belief)
            SCRIPT_FILE="room_comp/spoc_room_completion_3d_belief_workspace.py"
            SAVE_NAME="spoc_room_completion_3d_belief"
            EPISODE_OVERRIDE="${REPO_ROOT}/data/3d-core/room_completion"
            ;;
        obj_perm_3d_belief)
            SCRIPT_FILE="obj_perm/spoc_obj_permanence_3d_belief_workspace.py"
            SAVE_NAME="spoc_obj_permanence_3d_belief"
            EPISODE_OVERRIDE="${REPO_ROOT}/data/3d-core/object_permanence"
            ;;
        obj_comp_dfot_vggt)
            SCRIPT_FILE="obj_comp/spoc_obj_completion_dfot_vggt_workspace.py"
            SAVE_NAME="spoc_obj_completion_dfot_vggt"
            EPISODE_OVERRIDE="${REPO_ROOT}/data/3d-core/object_completion"
            ;;
        room_comp_dfot_vggt)
            SCRIPT_FILE="room_comp/spoc_room_completion_dfot_vggt_workspace.py"
            SAVE_NAME="spoc_room_completion_dfot_vggt"
            EPISODE_OVERRIDE="${REPO_ROOT}/data/3d-core/room_completion"
            ;;
        obj_perm_dfot_vggt)
            SCRIPT_FILE="obj_perm/spoc_obj_permanence_dfot_vggt_workspace.py"
            SAVE_NAME="spoc_obj_permanence_dfot_vggt"
            EPISODE_OVERRIDE="${REPO_ROOT}/data/3d-core/object_permanence"
            ;;
        *)
            echo "ERROR: Unknown agent '${agent}'."
            echo
            print_usage
            exit 1
            ;;
    esac
}

print_usage() {
    echo "Usage: bash $(basename "$0") <AGENT> [EXTRA_HYDRA_ARGS...]"
    echo
    echo "Available agents:"
    echo "  obj_comp_3d_belief       Object completion (3D belief)"
    echo "  room_comp_3d_belief      Room completion (3D belief)"
    echo "  obj_perm_3d_belief       Object permanence (3D belief)"
    echo "  obj_comp_dfot_vggt       Object completion (DFoT-VGGT)"
    echo "  room_comp_dfot_vggt      Room completion (DFoT-VGGT)"
    echo "  obj_perm_dfot_vggt       Object permanence (DFoT-VGGT)"
    echo
    echo "Extra Hydra overrides can be appended after the agent argument."
}

if [[ $# -lt 1 ]]; then
    print_usage
    exit 1
fi

AGENT="$1"
shift 1

EXTRA_HYDRA_ARGS=""
get_agent_config "${AGENT}"

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  set +u
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
else
  echo "[WARN] conda not found in PATH. Skipping conda activate."
fi

SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_FILE}"
SAVE_PATH="${OUTPUT_DIR}/${SAVE_NAME}"
EPISODE_ROOT="${EPISODE_OVERRIDE:-}"

echo "=============================================="
echo " Agent:       ${AGENT}"
echo " Script:      ${SCRIPT_PATH}"
echo " Save path:   ${SAVE_PATH}"
echo " Episode root: ${EPISODE_ROOT}"
echo "=============================================="

if [[ ! -f "${SCRIPT_PATH}" ]]; then
    echo "WARNING: Script '${SCRIPT_PATH}' does not exist."
    echo "         The command will likely fail. Continuing anyway..."
fi

HYDRA_FULL_ERROR=1 OC_CAUSE=1 python "${SCRIPT_PATH}" \
    +seed=42 \
    embodied_task.trajectory.save_path="${SAVE_PATH}" \
    embodied_task.episode_root="${EPISODE_ROOT}" \
    ${EXTRA_HYDRA_ARGS} \
    "$@"
