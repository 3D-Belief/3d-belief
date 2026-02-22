set -euo pipefail

# ================ Set up config parameters ====================
CONDA_ENV="3d-belief"
export XFORMERS_DISABLED=1
export OBJAVERSE_DATA_DIR="[Enter your objaverse data directory here]"
export OBJAVERSE_HOUSES_DIR="[Enter your objaverse houses directory here]"
# Base output directory (each baseline will create a subfolder here)
SAVE_ROOT="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs"
# SPOC trajectories root
EPISODE_ROOT="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test"
# Codebase root (used to build absolute paths below)
CODEBASE="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines"
# If you want to just print commands (no execution), set DRY_RUN=1
DRY_RUN="${DRY_RUN:-0}"
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
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
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
  "${HYDRA_ENV[@]}" "${cmd[@]}" 2>&1 | tee "$log_path"
}

NAV_WS="${CODEBASE}/belief_baselines/workspace/nav"

# Exploration
run_exp "spoc_obj_searching" \
  "${NAV_WS}/spoc_obj_searching_workspace.py"

# VLM Goal Selector
run_exp "spoc_obj_searching_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_vlm_goal_selector_workspace.py"

# GPT VLM Agent
run_exp "spoc_obj_searching_vlm_agent" \
  "${NAV_WS}/spoc_obj_searching_vlm_agent_workspace.py"

# Gemini VLM Agent
run_exp "spoc_obj_searching_gemini_vlm_agent" \
  "${NAV_WS}/spoc_obj_searching_gemini_vlm_agent_workspace.py"

# Finetuned VLM Agent
run_exp "spoc_obj_searching_qwen3_vlm_agent" \
  "${NAV_WS}/spoc_obj_searching_finetuned_vlm_agent_workspace.py"

# 3D Belief GPT VLM Goal Selector
run_exp "spoc_obj_searching_3d_belief_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_3d_belief_vlm_goal_selector_workspace.py"

# 3D Belief Gemini VLM Goal Selector
run_exp "spoc_obj_searching_3d_belief_gemini_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_3d_belief_gemini_vlm_goal_selector_workspace.py"

# VGGT Frontier Agent
run_exp "spoc_obj_searching_vggt_frontier" \
  "${NAV_WS}/spoc_obj_searching_vggt_frontier_workspace.py"

# VGGT VLM Goal Selector Agent
run_exp "spoc_obj_searching_vggt_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_vggt_vlm_goal_selector_workspace.py"

# VGGT Gemini VLM Goal Selector Agent
run_exp "spoc_obj_searching_vggt_gemini_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_vggt_gemini_vlm_goal_selector_workspace.py"

# 3D Belief Semantic Goal Selector Agent
run_exp "spoc_obj_searching_3d_belief_semantic_goal_selector_previous_weight" \
  "${NAV_WS}/spoc_obj_searching_3d_belief_semantic_goal_selector_workspace.py"

# DFoT-VGGT VLM Goal Selector Agent
run_exp "spoc_obj_searching_dfot_vggt_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_dfot_vggt_vlm_goal_selector_workspace.py"

# DFoT-VGGT Gemini VLM Goal Selector Agent
run_exp "spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector_workspace.py"

# NWM-VGGT VLM Goal Selector Agent
run_exp "spoc_obj_searching_nwm_vggt_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_nwm_vggt_vlm_goal_selector_workspace.py"

# NWM-VGGT Gemini VLM Goal Selector Agent
run_exp "spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector_workspace.py"

# Oracle Imagination Model with VLM Goal Selector
run_exp "spoc_obj_searching_oracle_imagination_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_oracle_imagination_vlm_goal_selector_workspace.py"

# Oracle Imagination Model with Gemini VLM Goal Selector
run_exp "spoc_obj_searching_oracle_imagination_gemini_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_oracle_imagination_gemini_vlm_goal_selector_workspace.py"

# 3D Belief No Imagination GPT VLM Goal Selector
run_exp "spoc_obj_searching_3d_belief_no_imagination_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_3d_belief_no_imagination_vlm_goal_selector_workspace.py"

# 3D Belief No Imagination Gemini VLM Goal Selector
run_exp "spoc_obj_searching_3d_belief_no_imagination_gemini_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_3d_belief_no_imagination_gemini_vlm_goal_selector_workspace.py"

# 3D Belief No 3D (CEM) GPT VLM Goal Selector
run_exp "spoc_obj_searching_3d_belief_cem_vlm_goal_selector" \
  "${NAV_WS}/spoc_obj_searching_3d_belief_cem_vlm_goal_selector_workspace.py"

# 3D Belief Frontier Agent
run_exp "spoc_obj_searching_3d_belief_frontier" \
  "${NAV_WS}/spoc_obj_searching_3d_belief_frontier_workspace.py"

echo ""
echo "[DONE] All short-subset baselines finished."
echo "Logs are in: ${LOG_DIR}"
