#!/bin/bash
#SBATCH --job-name=obj_search
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/tshu2/zwen19/3dbelief/3d-belief/wm_baselines/output/_logs/obj_search_%j.out
#SBATCH --error=/scratch/tshu2/zwen19/3dbelief/3d-belief/wm_baselines/output/_logs/obj_search_%j.err

set -euo pipefail

# ================ Set up config parameters ====================
CONDA_ENV="3d-belief"
export XFORMERS_DISABLED=1
export OBJAVERSE_DATA_DIR="/scratch/tshu2/zwen19/3dbelief/3d-belief/spoc_data/2023_07_28"
export OBJAVERSE_HOUSES_DIR="/scratch/tshu2/zwen19/3dbelief/3d-belief/spoc_data/houses_2023_07_28"
# Base output directory (each baseline will create a subfolder here)
SAVE_ROOT="/scratch/tshu2/zwen19/3dbelief/3d-belief/wm_baselines/output"
# SPOC trajectories root
EPISODE_ROOT="/scratch/tshu2/zwen19/3dbelief/3d-belief/spoc_data"
# Codebase root (used to build absolute paths below)
CODEBASE="/scratch/tshu2/zwen19/3dbelief/3d-belief"
# If you want to just print commands (no execution), set DRY_RUN=1
DRY_RUN="${DRY_RUN:-0}"
# Remember to export your OPENAI_API_KEY and GOOGLE_API_KEY in the environment for VLM-based agents
# =============================================================

if [[ ! -d "$EPISODE_ROOT" ]]; then
  echo "[ERROR] EPISODE_ROOT not found: $EPISODE_ROOT"
  exit 1
fi

mkdir -p "$SAVE_ROOT"
LOG_DIR="${SAVE_ROOT}/_logs"
mkdir -p "$LOG_DIR"

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

HYDRA_ENV=(HYDRA_FULL_ERROR=1 OC_CAUSE=1)

COMMON_OVERRIDES=(
  "embodied_task.episode_root=${EPISODE_ROOT}"
  "embodied_task.subset_type=length"
  "embodied_task.subset=short"
)

run_exp () {
  local exp_name="$1"
  local script_path="$2"
  shift 2
  local -a extra_overrides=("$@")

  if [[ ! -f "$script_path" ]]; then
    echo "[ERROR] Script not found: $script_path"
    exit 1
  fi

  local save_path="${SAVE_ROOT}/${exp_name}"
  local log_path="${LOG_DIR}/${exp_name}.log"

  mkdir -p "$save_path"

  local -a cmd=(
    python -u "$script_path"
    "embodied_task.trajectory.save_path=${save_path}"
    "${COMMON_OVERRIDES[@]}"
    "${extra_overrides[@]}"
  )

  echo ""
  echo "============================================================"
  echo "[RUN] ${exp_name}"
  echo "  script:   ${script_path}"
  echo "  save:     ${save_path}"
  echo "  episode:  ${EPISODE_ROOT}"
  echo "  log:      ${log_path}"
  echo "------------------------------------------------------------"
  printf '  %q ' "${HYDRA_ENV[@]}" "${cmd[@]}"
  echo ""
  echo "============================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY_RUN=1] Skipping execution."
    return 0
  fi

  # Run and tee logs
  env "${HYDRA_ENV[@]}" "${cmd[@]}" 2>&1 | tee "$log_path"
}

NAV_WS="${CODEBASE}/wm_baselines/workspace/nav"


# 3D Belief
run_exp "spoc_obj_searching_3d_belief_semantic_goal_selector_previous_weight" \
  "${NAV_WS}/spoc_obj_searching_3d_belief_semantic_goal_selector_workspace.py"

# GPT VLM Agent
run_exp "spoc_obj_searching_vlm_agent" \
  "${NAV_WS}/spoc_obj_searching_gpt_vlm_agent_workspace.py"

# Gemini VLM Agent
run_exp "spoc_obj_searching_gemini_vlm_agent" \
  "${NAV_WS}/spoc_obj_searching_gemini_vlm_agent_workspace.py"

# Qwen-3 VLM Agent
run_exp "spoc_obj_searching_qwen3_vlm_agent" \
  "${NAV_WS}/spoc_obj_searching_qwen3_vlm_agent_workspace.py"

# VGGT Frontier Agent
run_exp "spoc_obj_searching_vggt_frontier" \
  "${NAV_WS}/spoc_obj_searching_vggt_frontier_workspace.py"

# VGGT GPT VLM Goal Selector Agent
run_exp "spoc_obj_searching_vggt_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_vggt_gpt_vlm_goal_selector_workspace.py"

# VGGT Gemini VLM Goal Selector Agent
run_exp "spoc_obj_searching_vggt_gemini_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_vggt_gemini_vlm_goal_selector_workspace.py"

# DFoT-VGGT GPT VLM Goal Selector Agent
run_exp "spoc_obj_searching_dfot_vggt_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_dfot_vggt_gpt_vlm_goal_selector_workspace.py"

# DFoT-VGGT Gemini VLM Goal Selector Agent
run_exp "spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector_workspace.py"

# NWM-VGGT GPT VLM Goal Selector Agent
run_exp "spoc_obj_searching_nwm_vggt_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_nwm_vggt_gpt_vlm_goal_selector_workspace.py"

# NWM-VGGT Gemini VLM Goal Selector Agent
run_exp "spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector_workspace.py"

echo ""
echo "[DONE] All short-subset baselines finished."
echo "Logs are in: ${LOG_DIR}"
