# TODO: refactor so that it's easier to run
conda activate 3d-belief

export XFORMERS_DISABLED=1
export OBJAVERSE_DATA_DIR="[Enter your objaverse data directory here]"
export OBJAVERSE_HOUSES_DIR="[Enter your objaverse houses directory here]"

save_path = ""
episode_root = ""

# Exploration
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# VLM Goal Selector
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# GPT VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_gpt_vlm_agent_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_vlm_agent \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# GPT VLM Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_gpt_vlm_agent_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_vlm_agent_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# Gemini VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_gemini_vlm_agent_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_gemini_vlm_agent \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# Gemini VLM Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_gemini_vlm_agent_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_gemini_vlm_agent_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# Finetuned VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_qwen3_vlm_agent_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_qwen3_vlm_agent \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# Finetuned VLM Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_qwen3_vlm_agent_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_qwen3_vlm_agent_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# 3D Belief GPT VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief GPT VLM Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_vlm_goal_selector_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# 3D Belief Gemini VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_gemini_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_gemini_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief Gemini VLM Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_gemini_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_gemini_vlm_goal_selector_med_0_2 \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# VGGT Frontier Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_vggt_frontier_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_vggt_frontier \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# VGGT Frontier Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_vggt_frontier_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_vggt_frontier_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# VGGT VLM Goal Selector Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_vggt_gpt_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_vggt_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# VGGT Gemini VLM Goal Selector Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_vggt_gemini_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_vggt_gemini_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief Semantic Goal Selector Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_semantic_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_semantic_goal_selector_previous_weight \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief Semantic Goal Selector Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_semantic_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_semantic_goal_selector_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# DFoT-VGGT VLM Goal Selector Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_dfot_vggt_gpt_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_dfot_vggt_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# DFoT-VGGT VLM Goal Selector Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_dfot_vggt_gpt_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_dfot_vggt_vlm_goal_selector_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# DFoT-VGGT Gemini VLM Goal Selector Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# DFoT-VGGT Gemini VLM Goal Selector Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_dfot_vggt_gemini_vlm_goal_selector_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# NWM-VGGT VLM Goal Selector Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_nwm_vggt_gpt_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_nwm_vggt_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# NWM-VGGT VLM Goal Selector Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_nwm_vggt_gpt_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_nwm_vggt_vlm_goal_selector_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# NWM-VGGT Gemini VLM Goal Selector Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# NWM-VGGT Gemini VLM Goal Selector Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_nwm_gemini_vggt_vlm_goal_selector_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# Oracle Imagination Model with VLM Goal Selector
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_oracle_imagination_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_oracle_imagination_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# Oracle Imagination Model with Gemini VLM Goal Selector
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_oracle_imagination_gemini_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_oracle_imagination_gemini_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief No Imagination GPT VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_no_imagination_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_no_imagination_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief No Imagination Gemini VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_no_imagination_gemini_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_no_imagination_gemini_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief No 3D GPT VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_cem_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_cem_vlm_goal_selector \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief Frontier Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_frontier_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_frontier \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \

# 3D Belief Frontier Agent - Medium
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_frontier_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_frontier_med \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=med \

# (Demo) Gemini VLM Agent
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/belief_baselines/workspace/nav/spoc_obj_searching_3d_belief_gemini_vlm_goal_selector_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/spoc_obj_searching_3d_belief_gemini_vlm_goal_selector_demo \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_trajectories/test \
    embodied_task.subset_type=length \
    embodied_task.subset=short \
    embodied_task.timeout=150 \
    agent.world_model.fast_sampling=false \
    agent.world_model.model.diffusion_model.temperature=0.8 \