#!/bin/bash
# Training: s3db_dataset, Dense Layout + SG + Recon + FiLM + MiniLM Layout Loss
#
# Same as train_s3db_dense_clip.sh but uses all-MiniLM-L6-v2 (384d) text
# embeddings instead of CLIP ViT-B/32 (512d) for layout conditioning.
set -e

# ---- paths ----
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_ROOT="${REPO_ROOT}/../datasets/s3db_dataset"
VOCAB_DIR="${REPO_ROOT}/outputs/vocab/s3db"
EMBEDDINGS_PATH="${VOCAB_DIR}/sg_type_embeddings_minilm.pt"

# ---- environment ----
eval "$(conda shell.bash hook)"
conda activate 3d-belief

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Ensure local repo is found before any pip-installed copy
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=4
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
export MASTER_PORT=$((29500 + RANDOM % 1000))

nvidia-smi

# Number of object types (from s3db vocab)
N_TYPES=750

# ---- Train ----
torchrun --nnodes 1 --nproc_per_node 1 --master_port $MASTER_PORT \
    splat_belief/experiment/train.py \
    dataset=s3db \
    dataset.root_dir="${DATASET_ROOT}" \
    dataset.vocab_dir="${VOCAB_DIR}" \
    dataset.vggt_alignment_loss_weight=2.0 \
    dataset.intermediate_weight=5.0 \
    dataset.depth_smooth_loss_weight=0.1 \
    dataset.layout_recon_loss_weight=1.0 \
    dataset.dense_clip_layout_loss_weight=1.0 \
    dataset.semantic_loss_weight=0.0 \
    dataset.include_walls=true \
    dataset.wall_height_default=2.5 \
    dataset.wall_thickness=0.15 \
    setting_name=pixelsplat_h100 \
    stage=train \
    results_folder=outputs/training/s3db_dense_minilm_lock \
    semantic_config=${REPO_ROOT}/splat_belief/config/semantic/onehot.yaml \
    checkpoint_path=${REPO_ROOT}/outputs/training/s3db_base/model-22.pt \
    ngpus=1 \
    image_size=128 \
    ctxt_min=5 \
    ctxt_max=15 \
    model/encoder=uvitmvsplat_sg \
    model.encoder.use_image_condition=true \
    model.encoder.depth_predictor_time_embed=true \
    model.encoder.use_camera_pose=true \
    model.encoder.use_semantic=true \
    model.encoder.use_reg_model=false \
    model.encoder.d_semantic=384 \
    model.encoder.d_semantic_reg=384 \
    model.encoder.gaussians_per_pixel=1 \
    model.encoder.evolve_ctxt=false \
    model.encoder.use_depth_mask=true \
    model.encoder.freeze_depth_predictor=false \
    model/encoder/backbone=u_vit3d_pose_sg \
    model.encoder.backbone.sg_type_embeddings_path="${EMBEDDINGS_PATH}" \
    model.encoder.backbone.sg_text_encoder=minilm \
    model.encoder.backbone.sg_clip_dim=384 \
    model.encoder.backbone.use_vggt_alignment=true \
    model.encoder.backbone.use_repa=true \
    model.encoder.backbone.input_size='[128, 128]' \
    model.encoder.backbone.sg_use_gcn=false \
    model.encoder.backbone.sg_spatial_mode=bbox_surface \
    model.encoder.backbone.n_object_types=${N_TYPES} \
    model.encoder.backbone.include_walls=true \
    model.encoder.backbone.use_dense_layout=true \
    model.encoder.backbone.layout_embed_dim=256 \
    model.encoder.backbone.layout_injection_mode=film \
    model.encoder.backbone.use_sparse_sg=false \
    model.encoder.backbone.use_layout_recon_loss=true \
    alignment.latents_info=-1 \
    ctxt_losses_factor=0.7 \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    name=s3db_dense_minilm_lock \
    wandb=local \
    clean_target=false \
    use_identity=true \
    intermediate=true \
    load_optimizer=false \
    load_enc=false \
    lock_enc_steps=4000 \
    use_depth_smoothness=true \
    adjacent_angle=0.785 \
    adjacent_distance=1.0 \
    num_intermediate=15
