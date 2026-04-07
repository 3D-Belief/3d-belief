#!/bin/bash
# Evaluate 3D-Belief with and without online Gaussian refinement.
# Runs temporal_inference.py twice on poc_dataset/train:
#   1) Baseline (refiner off)
#   2) Refined  (refiner on, 60 Adam iterations per keyframe step)
# Produces: PLY scenes, RGB/depth/semantic videos, metrics.json per scene.
# Then run compare_refinement.py on the two output dirs to get a table.

set -e

# ---- Configuration (override via environment variables) ----
NUM_SAMPLES=${NUM_SAMPLES:-5}
SEED=${SEED:-42}
DATASET_ROOT=${DATASET_ROOT:-/home/ubuntu/tianmin-neurips/yyin34/codebase/structured_3d_belief/datasets/poc_dataset}
CHECKPOINT=${CHECKPOINT:-/home/ubuntu/tianmin-neurips/yyin34/codebase/structured_3d_belief/3d-belief/checkpoints/3d_belief_spoc.pt}
BASELINE_DIR=${BASELINE_DIR:-outputs/eval/poc_baseline}
REFINED_DIR=${REFINED_DIR:-outputs/eval/poc_refined}

# ---- Environment ----
eval "$(conda shell.bash hook)"
conda activate 3d-belief

export PYTHONPATH="$PWD:$PWD/splat_belief:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((12000 + RANDOM % 1000))
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

nvidia-smi

# ---- Common flags ----
COMMON_FLAGS=" \
    dataset=spoc_seq \
    dataset.root_dir=${DATASET_ROOT} \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    stage=train \
    model/encoder=uvitmvsplat \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=true \
    model.encoder.use_reg_model=true \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=384 \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.inference_mode=false \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.input_size='[128, 128]' \
    model_type=uvit_pose \
    semantic_mode=embed \
    semantic_viz=query \
    semantic_config=splat_belief/config/semantic/onehot.yaml \
    temperature=0.85 \
    sampling_steps=50 \
    image_size=128 \
    inference_sample_from_dataset=true \
    inference_num_samples=${NUM_SAMPLES} \
    inference_min_frames=5 \
    inference_max_frames=25 \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    clean_target=false \
    use_history=false \
    inference_save_scene=false \
    pose_source=gt \
    checkpoint_path=${CHECKPOINT} \
    seed=${SEED} \
"

echo "============================================================"
echo " Run 1/2: BASELINE (refiner OFF)"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    ${COMMON_FLAGS} \
    refiner.enabled=false \
    results_folder=${BASELINE_DIR} \
    name=eval_baseline

echo ""
echo "============================================================"
echo " Run 2/2: REFINED (refiner ON, 30 iterations)"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    ${COMMON_FLAGS} \
    refiner.enabled=true \
    refiner.num_iterations=30 \
    results_folder=${REFINED_DIR} \
    name=eval_refined

echo ""
echo "============================================================"
echo " Comparing results"
echo "============================================================"
python scripts/calculate_metrics/compare_refinement.py \
    ${BASELINE_DIR} ${REFINED_DIR}

echo "Done."
conda deactivate