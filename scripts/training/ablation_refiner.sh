#!/bin/bash
# Ablation: compare 4 refiner configurations on the same scene.
#   1) Baseline        — refiner OFF
#   2) No-reg          — refiner ON, no regularization, no visibility mask
#   3) Reg-only        — refiner ON, with regularization, no visibility mask
#   4) Reg + Vis-mask  — refiner ON, with regularization + visibility masking

set -e

SEED=${SEED:-42}
DATASET_ROOT=${DATASET_ROOT:-/home/ubuntu/tianmin-neurips/yyin34/codebase/structured_3d_belief/datasets/poc_dataset}
CHECKPOINT=${CHECKPOINT:-/home/ubuntu/tianmin-neurips/yyin34/codebase/structured_3d_belief/3d-belief/checkpoints/3d_belief_spoc.pt}
OUT_BASE=${OUT_BASE:-outputs/eval/ablation}

eval "$(conda shell.bash hook)"
conda activate 3d-belief

export PYTHONPATH="$PWD:$PWD/splat_belief:$PYTHONPATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MASTER_PORT=$((12000 + RANDOM % 1000))
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

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
    model.encoder.backbone.input_size=[128,128] \
    model_type=uvit_pose \
    semantic_mode=embed \
    semantic_viz=query \
    semantic_config=splat_belief/config/semantic/onehot.yaml \
    temperature=0.85 \
    sampling_steps=50 \
    image_size=128 \
    inference_sample_from_dataset=false \
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
echo " Run 1/4: BASELINE (refiner OFF)"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    ${COMMON_FLAGS} \
    refiner.enabled=false \
    results_folder=${OUT_BASE}/baseline \
    name=ablation_baseline

echo ""
echo "============================================================"
echo " Run 2/4: NO REGULARIZATION, NO VISIBILITY MASK"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    ${COMMON_FLAGS} \
    refiner.enabled=true \
    refiner.num_iterations=30 \
    refiner.prior_weight=0.0 \
    refiner.isotropic_weight=0.0 \
    refiner.use_visibility_masking=false \
    refiner.freeze_geometry=false \
    refiner.use_lr_decay=false \
    refiner.min_observations=2 \
    results_folder=${OUT_BASE}/no_reg_no_vis \
    name=ablation_no_reg_no_vis

echo ""
echo "============================================================"
echo " Run 3/4: WITH REGULARIZATION, NO VISIBILITY MASK"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    ${COMMON_FLAGS} \
    refiner.enabled=true \
    refiner.num_iterations=30 \
    refiner.prior_weight=0.1 \
    refiner.isotropic_weight=10.0 \
    refiner.use_visibility_masking=false \
    refiner.freeze_geometry=false \
    refiner.use_lr_decay=true \
    refiner.min_observations=2 \
    results_folder=${OUT_BASE}/reg_no_vis \
    name=ablation_reg_no_vis

echo ""
echo "============================================================"
echo " Run 4/4: WITH REGULARIZATION + VISIBILITY MASK"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    ${COMMON_FLAGS} \
    refiner.enabled=true \
    refiner.num_iterations=30 \
    refiner.prior_weight=0.1 \
    refiner.isotropic_weight=10.0 \
    refiner.use_visibility_masking=true \
    refiner.freeze_geometry=false \
    refiner.use_lr_decay=true \
    refiner.min_observations=2 \
    results_folder=${OUT_BASE}/reg_vis \
    name=ablation_reg_vis

echo ""
echo "============================================================"
echo " Comparing all results"
echo "============================================================"
python -c "
import json, glob, os, sys
import numpy as np

base = '${OUT_BASE}'
configs = {
    'baseline':       'baseline',
    'no_reg_no_vis':  'no_reg_no_vis',
    'reg_no_vis':     'reg_no_vis',
    'reg_vis':        'reg_vis',
}

results = {}
for label, folder in configs.items():
    pattern = os.path.join(base, folder, 'visuals_*', 'metrics.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'WARNING: no metrics found for {label} at {pattern}')
        continue
    m = json.load(open(files[0]))
    results[label] = m

if not results:
    print('No results found!')
    sys.exit(1)

# Print comparison table
metrics = ['mean_psnr', 'mean_ssim', 'mean_lpips', 'mean_l1',
           'keyframe_mean_psnr', 'keyframe_mean_ssim', 'keyframe_mean_lpips',
           'nonkeyframe_mean_psnr', 'nonkeyframe_mean_ssim', 'nonkeyframe_mean_lpips']

header = f\"{'Metric':<30s}\"
for label in results:
    header += f'{label:>18s}'
print(header)
print('-' * len(header))

for metric in metrics:
    row = f'{metric:<30s}'
    for label in results:
        val = results[label].get(metric, float('nan'))
        row += f'{val:>18.4f}'
    print(row)

# Delta table vs baseline
if 'baseline' in results:
    print()
    print('=== Delta vs Baseline ===')
    header2 = f\"{'Metric':<30s}\"
    for label in results:
        if label == 'baseline': continue
        header2 += f'{label:>18s}'
    print(header2)
    print('-' * len(header2))
    for metric in metrics:
        row = f'{metric:<30s}'
        base_val = results['baseline'].get(metric, 0)
        for label in results:
            if label == 'baseline': continue
            val = results[label].get(metric, float('nan'))
            delta = val - base_val
            sign = '+' if delta >= 0 else ''
            row += f'{sign}{delta:>17.4f}'
        print(row)

# Per-frame comparison
print()
print('=== Per-frame PSNR ===')
header3 = f\"{'Frame':<8s}{'KF?':<6s}\"
for label in results:
    header3 += f'{label:>18s}'
print(header3)
print('-' * len(header3))
base_pf = results.get('baseline', {}).get('per_frame', [])
n_frames = len(base_pf) if base_pf else 0
for fi in range(n_frames):
    is_kf = base_pf[fi]['is_keyframe']
    row = f'{fi:<8d}{\"Y\" if is_kf else \"\":<6s}'
    for label in results:
        pf = results[label].get('per_frame', [])
        if fi < len(pf):
            row += f'{pf[fi][\"psnr\"]:>18.2f}'
        else:
            row += f'{\"N/A\":>18s}'
    print(row)
"

echo "Done."
