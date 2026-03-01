# ==============================================================================
# Object Searching Rollout Script
#
# Usage:
#   bash object_searching.sh <AGENT> [EXTRA_ARGS...]
#
# Arguments:
#   AGENT      - Agent name (see list below, required)
#   EXTRA_ARGS - Any additional Hydra overrides passed through to the command
#
# Examples:
#   bash object_searching.sh gemini_vlm_agent
#   bash object_searching.sh gpt_vlm_agent
#   bash object_searching.sh vggt_frontier embodied_task.timeout=150
#
# ==============================================================================

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────
# Root of this repository
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Directory containing workspace Python scripts
SCRIPT_DIR="${REPO_ROOT}/wm_baselines/workspace/nav"
# Base output directory for saving trajectories
OUTPUT_DIR="${REPO_ROOT}/outputs"
# Episode root directory (SPOC test trajectories)
EPISODE_ROOT=/scratch/tshu2/zwen19/3dbelief/3d-belief/data/spoc_trajectories_val
# Conda environment name
CONDA_ENV="3d-belief"
# Environment variables
export XFORMERS_DISABLED=1
export OBJAVERSE_DATA_DIR="/scratch/tshu2/zwen19/3dbelief/3d-belief/spoc_data/2023_07_28"
export OBJAVERSE_HOUSES_DIR="/scratch/tshu2/zwen19/3dbelief/3d-belief/spoc_data/houses_2023_07_28"

# ── Available Agents ──────────────────────────────────────────────────────────
#
#  Agent Key                                | Description
#  -----------------------------------------+-------------------------------------------
#  3d_belief_semantic_goal_selector         | 3D Belief + Semantic Goal Selector
#  gpt_vlm_agent                            | GPT VLM Agent
#  gemini_vlm_agent                         | Gemini VLM Agent
#  qwen3_vlm_agent                      | Finetuned (Qwen3) VLM Agent
#  vggt_frontier                            | VGGT + Frontier Agent
#  vggt_gpt_vlm_goal_selector            | VGGT + GPT VLM Goal Selector
#  vggt_gemini_vlm_goal_selector           | VGGT + Gemini VLM Goal Selector
#  dfot_vggt_gpt_vlm_goal_selector         | DFoT-VGGT + GPT VLM Goal Selector
#  dfot_vggt_gemini_vlm_goal_selector      | DFoT-VGGT + Gemini VLM Goal Selector
#  nwm_vggt_gpt_vlm_goal_selector          | NWM-VGGT + GPT VLM Goal Selector
#  nwm_vggt_gemini_vlm_goal_selector       | NWM-VGGT + Gemini VLM Goal Selector
#

get_agent_config() {
    local agent="$1"
    # Sets SCRIPT_FILE and SAVE_NAME
    case "${agent}" in
        3d_belief_semantic_goal_selector)
            SCRIPT_FILE="spoc_obj_searching_3d_belief_semantic_goal_selector_workspace.py"
            SAVE_NAME="spoc_obj_searching_3d_belief_semantic_goal_selector_previous_weight"
            ;;
        gpt_vlm_agent)
            SCRIPT_FILE="spoc_obj_searching_gpt_vlm_agent_workspace.py"
            SAVE_NAME="spoc_obj_searching_gpt_vlm_agent"
            ;;
        gemini_vlm_agent)
            SCRIPT_FILE="spoc_obj_searching_gemini_vlm_agent_workspace.py"
            SAVE_NAME="spoc_obj_searching_gemini_vlm_agent"
            ;;
        qwen3_vlm_agent)
            SCRIPT_FILE="spoc_obj_searching_qwen3_vlm_agent_workspace.py"
            SAVE_NAME="spoc_obj_searching_qwen3_vlm_agent"
            ;;
        vggt_frontier)
            SCRIPT_FILE="spoc_obj_searching_vggt_frontier_workspace.py"
            SAVE_NAME="spoc_obj_searching_vggt_frontier"
            ;;
        vggt_gpt_vlm_goal_selector)
            SCRIPT_FILE="spoc_obj_searching_vggt_gpt_vlm_goal_selector_workspace.py"
            SAVE_NAME="spoc_obj_searching_vggt_gpt_vlm_goal_selector"
            ;;
        vggt_gemini_vlm_goal_selector)
            SCRIPT_FILE="spoc_obj_searching_vggt_gemini_vlm_goal_selector_workspace.py"
            SAVE_NAME="spoc_obj_searching_vggt_gemini_vlm_goal_selector"
            ;;
        dfot_vggt_gpt_vlm_goal_selector)
            SCRIPT_FILE="spoc_obj_searching_dfot_vggt_gpt_vlm_goal_selector_workspace.py"
            SAVE_NAME="spoc_obj_searching_dfot_vggt_gpt_vlm_goal_selector"
            ;;
        dfot_vggt_gemini_vlm_goal_selector)
            SCRIPT_FILE="spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector_workspace.py"
            SAVE_NAME="spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector"
            ;;
        nwm_vggt_gpt_vlm_goal_selector)
            SCRIPT_FILE="spoc_obj_searching_nwm_vggt_gpt_vlm_goal_selector_workspace.py"
            SAVE_NAME="spoc_obj_searching_nwm_vggt_gpt_vlm_goal_selector"
            ;;
        nwm_vggt_gemini_vlm_goal_selector)
            SCRIPT_FILE="spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector_workspace.py"
            SAVE_NAME="spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector"
            ;;
        *)
            echo "ERROR: Unknown agent '${agent}'."
            echo ""
            print_usage
            exit 1
            ;;
    esac
}

print_usage() {
    echo "Usage: bash $(basename "$0") <AGENT> [EXTRA_ARGS...]"
    echo ""
    echo "Available agents:"
    echo "  3d_belief_semantic_goal_selector        3D Belief + Semantic Goal Selector"
    echo "  gpt_vlm_agent                           GPT VLM Agent"
    echo "  gemini_vlm_agent                        Gemini VLM Agent"
    echo "  finetuned_vlm_agent                     Finetuned (Qwen3) VLM Agent"
    echo "  vggt_frontier                           VGGT + Frontier Agent"
    echo "  vggt_gpt_vlm_goal_selector              VGGT + GPT VLM Goal Selector"
    echo "  vggt_gemini_vlm_goal_selector           VGGT + Gemini VLM Goal Selector"
    echo "  dfot_vggt_gpt_vlm_goal_selector         DFoT-VGGT + GPT VLM Goal Selector"
    echo "  dfot_vggt_gemini_vlm_goal_selector      DFoT-VGGT + Gemini VLM Goal Selector"
    echo "  nwm_vggt_gpt_vlm_goal_selector          NWM-VGGT + GPT VLM Goal Selector"
    echo "  nwm_vggt_gemini_vlm_goal_selector       NWM-VGGT + Gemini VLM Goal Selector"
    echo ""
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
  set +u  # conda scripts reference unset variables
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
  set -u
  # Ensure conda's libstdc++ is found before the (older) system one
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
else
  echo "[WARN] conda not found in PATH. Skipping conda activate."
fi

SCRIPT_PATH="${SCRIPT_DIR}/${SCRIPT_FILE}"
SAVE_PATH="${OUTPUT_DIR}/${SAVE_NAME}"

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
    embodied_task.trajectory.save_path="${SAVE_PATH}" \
    embodied_task.episode_root="${EPISODE_ROOT}" \
    embodied_task.subset_type=length \
    embodied_task.subset=short \
    ${EXTRA_HYDRA_ARGS} \
    "$@"
