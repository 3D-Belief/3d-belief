#!/bin/bash
# Ablation evaluation: compare 4 ProcTHOR model variants with layout recon loss.
#   1) Base                        — no SG, no dense layout  (model-53)
#   2) Dense Layout + Recon        — dense layout, recon loss, lock 5K  (model-13)
#   3) Dense Layout + Recon NoLock — dense layout, recon loss, no lock  (model-13)
#   4) Dense Layout + SG + Recon   — dense layout + SG, recon loss, lock 5K  (model-12)
set -e

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${REPO_ROOT}/../datasets/poc_dataset"
VOCAB_DIR="${REPO_ROOT}/outputs/vocab/procthor"
EMBEDDINGS_PATH="${VOCAB_DIR}/sg_type_embeddings.pt"
TRAIN_DIR="${REPO_ROOT}/outputs/training"
OUT_BASE="${REPO_ROOT}/outputs/eval/procthor_recon_ablation"

eval "$(conda shell.bash hook)"
conda activate 3d-belief

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export CUDA_VISIBLE_DEVICES=1

SEED=42

COMMON_INPUT_SIZE="model.encoder.backbone.input_size=[128,128]"

echo "============================================================"
echo " Run 1/4: BASE (no SG, no dense layout)"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    dataset=procthor \
    dataset.root_dir=${DATASET_ROOT} \
    dataset.vocab_dir=${VOCAB_DIR} \
    stage=test \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    image_size=128 \
    model/encoder=uvitmvsplat \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=false \
    model.encoder.use_reg_model=false \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=384 \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_depth_mask=false \
    model.encoder.inference_mode=false \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.use_vggt_alignment=true \
    model.encoder.backbone.use_repa=true \
    "${COMMON_INPUT_SIZE}" \
    alignment.latents_info=-1 \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    semantic_config=configurations/semantic/onehot.yaml \
    wandb=local \
    clean_target=false \
    use_identity=true \
    intermediate=false \
    use_depth_smoothness=true \
    use_history=false \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    temperature=0.85 \
    sampling_steps=50 \
    inference_sample_from_dataset=false \
    inference_num_samples=1 \
    inference_min_frames=15 \
    inference_max_frames=20 \
    dataset.pose_source=gt \
    seed=${SEED} \
    ctxt_min=5 \
    ctxt_max=15 \
    checkpoint_path=${TRAIN_DIR}/procthor_base_weights/model-53.pt \
    results_folder=${OUT_BASE}/base \
    name=eval_base

echo ""
echo "============================================================"
echo " Run 2/4: DENSE LAYOUT + RECON LOSS (lock 5K)"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    dataset=procthor \
    dataset.root_dir=${DATASET_ROOT} \
    dataset.vocab_dir=${VOCAB_DIR} \
    dataset.include_walls=true \
    dataset.wall_height_default=2.5 \
    dataset.wall_thickness=0.15 \
    stage=test \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    image_size=128 \
    model/encoder=uvitmvsplat_sg \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=false \
    model.encoder.use_reg_model=false \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=384 \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_depth_mask=false \
    model.encoder.inference_mode=false \
    model.encoder.freeze_depth_predictor=false \
    model/encoder/backbone=u_vit3d_pose_sg \
    model.encoder.backbone.sg_type_embeddings_path=${EMBEDDINGS_PATH} \
    model.encoder.backbone.use_vggt_alignment=true \
    model.encoder.backbone.use_repa=true \
    model.encoder.backbone.sg_use_gcn=true \
    model.encoder.backbone.sg_spatial_mode=bbox_surface \
    model.encoder.backbone.n_object_types=204 \
    model.encoder.backbone.include_walls=true \
    model.encoder.backbone.use_dense_layout=true \
    model.encoder.backbone.layout_embed_dim=128 \
    model.encoder.backbone.use_sparse_sg=false \
    model.encoder.backbone.use_layout_recon_loss=true \
    "${COMMON_INPUT_SIZE}" \
    alignment.latents_info=-1 \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    semantic_config=configurations/semantic/onehot.yaml \
    wandb=local \
    clean_target=false \
    use_identity=true \
    intermediate=false \
    use_depth_smoothness=true \
    use_history=false \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    temperature=0.85 \
    sampling_steps=50 \
    inference_sample_from_dataset=false \
    inference_num_samples=1 \
    inference_min_frames=15 \
    inference_max_frames=20 \
    dataset.pose_source=gt \
    seed=${SEED} \
    ctxt_min=5 \
    ctxt_max=15 \
    checkpoint_path=${TRAIN_DIR}/procthor_dense_layout_recon/model-13.pt \
    results_folder=${OUT_BASE}/dense_layout_recon \
    name=eval_dense_layout_recon

echo ""
echo "============================================================"
echo " Run 3/4: DENSE LAYOUT + RECON LOSS (no lock)"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    dataset=procthor \
    dataset.root_dir=${DATASET_ROOT} \
    dataset.vocab_dir=${VOCAB_DIR} \
    dataset.include_walls=true \
    dataset.wall_height_default=2.5 \
    dataset.wall_thickness=0.15 \
    stage=test \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    image_size=128 \
    model/encoder=uvitmvsplat_sg \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=false \
    model.encoder.use_reg_model=false \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=384 \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_depth_mask=false \
    model.encoder.inference_mode=false \
    model.encoder.freeze_depth_predictor=false \
    model/encoder/backbone=u_vit3d_pose_sg \
    model.encoder.backbone.sg_type_embeddings_path=${EMBEDDINGS_PATH} \
    model.encoder.backbone.use_vggt_alignment=true \
    model.encoder.backbone.use_repa=true \
    model.encoder.backbone.sg_use_gcn=true \
    model.encoder.backbone.sg_spatial_mode=bbox_surface \
    model.encoder.backbone.n_object_types=204 \
    model.encoder.backbone.include_walls=true \
    model.encoder.backbone.use_dense_layout=true \
    model.encoder.backbone.layout_embed_dim=128 \
    model.encoder.backbone.use_sparse_sg=false \
    model.encoder.backbone.use_layout_recon_loss=true \
    "${COMMON_INPUT_SIZE}" \
    alignment.latents_info=-1 \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    semantic_config=configurations/semantic/onehot.yaml \
    wandb=local \
    clean_target=false \
    use_identity=true \
    intermediate=false \
    use_depth_smoothness=true \
    use_history=false \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    temperature=0.85 \
    sampling_steps=50 \
    inference_sample_from_dataset=false \
    inference_num_samples=1 \
    inference_min_frames=15 \
    inference_max_frames=20 \
    dataset.pose_source=gt \
    seed=${SEED} \
    ctxt_min=5 \
    ctxt_max=15 \
    checkpoint_path=${TRAIN_DIR}/procthor_dense_layout_recon_no_lock/model-13.pt \
    results_folder=${OUT_BASE}/dense_layout_recon_no_lock \
    name=eval_dense_layout_recon_no_lock

echo ""
echo "============================================================"
echo " Run 4/4: DENSE LAYOUT + SG + RECON LOSS (lock 5K)"
echo "============================================================"
python splat_belief/experiment/temporal_inference.py \
    dataset=procthor \
    dataset.root_dir=${DATASET_ROOT} \
    dataset.vocab_dir=${VOCAB_DIR} \
    dataset.include_walls=true \
    dataset.wall_height_default=2.5 \
    dataset.wall_thickness=0.15 \
    stage=test \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    image_size=128 \
    model/encoder=uvitmvsplat_sg \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=false \
    model.encoder.use_reg_model=false \
    model.encoder.d_semantic=512 \
    model.encoder.d_semantic_reg=384 \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_depth_mask=false \
    model.encoder.inference_mode=false \
    model.encoder.freeze_depth_predictor=false \
    model/encoder/backbone=u_vit3d_pose_sg \
    model.encoder.backbone.sg_type_embeddings_path=${EMBEDDINGS_PATH} \
    model.encoder.backbone.use_vggt_alignment=true \
    model.encoder.backbone.use_repa=true \
    model.encoder.backbone.sg_use_gcn=true \
    model.encoder.backbone.sg_spatial_mode=bbox_surface \
    model.encoder.backbone.n_object_types=204 \
    model.encoder.backbone.include_walls=true \
    model.encoder.backbone.use_dense_layout=true \
    model.encoder.backbone.layout_embed_dim=128 \
    model.encoder.backbone.use_sparse_sg=true \
    model.encoder.backbone.use_layout_recon_loss=true \
    "${COMMON_INPUT_SIZE}" \
    alignment.latents_info=-1 \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    semantic_config=configurations/semantic/onehot.yaml \
    wandb=local \
    clean_target=false \
    use_identity=true \
    intermediate=false \
    use_depth_smoothness=true \
    use_history=false \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    temperature=0.85 \
    sampling_steps=50 \
    inference_sample_from_dataset=false \
    inference_num_samples=1 \
    inference_min_frames=15 \
    inference_max_frames=20 \
    dataset.pose_source=gt \
    seed=${SEED} \
    ctxt_min=5 \
    ctxt_max=15 \
    checkpoint_path=${TRAIN_DIR}/procthor_dense_layout_sg_recon/model-12.pt \
    results_folder=${OUT_BASE}/dense_layout_sg_recon \
    name=eval_dense_layout_sg_recon

echo ""
echo "============================================================"
echo " Comparing all results"
echo "============================================================"
python -c "
import json, glob, os, sys

base = '${OUT_BASE}'
configs = {
    'base':                    'base',
    'dense_layout_recon':      'dense_layout_recon',
    'dense_layout_recon_nolock': 'dense_layout_recon_no_lock',
    'dense_layout_sg_recon':   'dense_layout_sg_recon',
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

header = f\"{'Metric':<35s}\"
for label in results:
    header += f'{label:>25s}'
print(header)
print('-' * len(header))

for metric in metrics:
    row = f'{metric:<35s}'
    for label in results:
        val = results[label].get(metric, float('nan'))
        row += f'{val:>25.4f}'
    print(row)

# Delta table vs base
if 'base' in results:
    print()
    print('=== Delta vs Base ===')
    header2 = f\"{'Metric':<35s}\"
    for label in results:
        if label == 'base': continue
        header2 += f'{label:>25s}'
    print(header2)
    print('-' * len(header2))
    for metric in metrics:
        row = f'{metric:<35s}'
        base_val = results['base'].get(metric, 0)
        for label in results:
            if label == 'base': continue
            val = results[label].get(metric, float('nan'))
            delta = val - base_val
            sign = '+' if delta >= 0 else ''
            row += f'{sign}{delta:>24.4f}'
        print(row)

# Per-frame PSNR if available
print()
print('=== Per-frame PSNR ===')
header3 = f\"{'Frame':<8s}{'KF?':<6s}\"
for label in results:
    header3 += f'{label:>25s}'
print(header3)
print('-' * len(header3))
first_label = list(results.keys())[0]
base_pf = results[first_label].get('per_frame', [])
for fi in range(len(base_pf)):
    is_kf = base_pf[fi].get('is_keyframe', False)
    row = f'{fi:<8d}{\"Y\" if is_kf else \"\":<6s}'
    for label in results:
        pf = results[label].get('per_frame', [])
        if fi < len(pf):
            row += f'{pf[fi][\"psnr\"]:>25.2f}'
        else:
            row += f'{\"N/A\":>25s}'
    print(row)
"

echo "Done."
