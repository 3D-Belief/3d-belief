source /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/miniconda3/etc/profile.d/conda.sh
conda activate belief

export XFORMERS_DISABLED=1
export CUDA_VISIBLE_DEVICES=4
export OBJAVERSE_DATA_DIR="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data"
export OBJAVERSE_HOUSES_DIR="/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/houses_2023_07_28"

# object completion with 3D belief
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/wm_baselines/workspace/obj_comp/spoc_obj_completion_3d_belief_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/outputs/spoc_obj_completion_3d_belief_T1.2_filtered \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_object_visibility_filtered

# room completion with 3D belief
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/wm_baselines/workspace/room_comp/spoc_room_completion_3d_belief_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/outputs/spoc_room_completion_3d_belief \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_door_passing

# object permanence with 3D belief
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/wm_baselines/workspace/obj_perm/spoc_obj_permanence_3d_belief_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/outputs/spoc_obj_permanence_3d_belief \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_full_rotation_unit

# object completion with DFoT-VGGT
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/wm_baselines/workspace/obj_comp/spoc_obj_completion_dfot_vggt_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/outputs/spoc_obj_completion_dfot_vggt_filtered \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_object_visibility_filtered

# room completion with DFoT-VGGT
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/wm_baselines/workspace/room_comp/spoc_room_completion_dfot_vggt_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/outputs/spoc_room_completion_dfot_vggt \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_door_passing

# object permanence with DFoT-VGGT
HYDRA_FULL_ERROR=1 OC_CAUSE=1 python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/wm_baselines/workspace/obj_perm/spoc_obj_permanence_dfot_vggt_workspace.py \
    embodied_task.trajectory.save_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/outputs/spoc_obj_permanence_dfot_vggt \
    embodied_task.episode_root=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/spoc/data/spoc_full_rotation_unit
