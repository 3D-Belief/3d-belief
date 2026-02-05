source uv_venv/.venv/bin/activate

nvidia-smi

export MASTER_PORT=$((12000 + RANDOM % 1000))

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

CUDA_LAUNCH_BLOCKING=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port $MASTER_PORT\
    splat_belief/experiment/train.py \
    dataset=spoc \
    dataset.vggt_alignment_loss_weight=2.0 \
    dataset.intermediate_weight=5.0 \
    dataset.depth_smooth_loss_weight=0.1 \
    dataset.root_dir=/path/to/dataset \
    setting_name=pixelsplat_h100 \
    stage=unit \
    results_folder=outputs/training/spoc_base \
    semantic_config=configurations/semantic/onehot.yaml \
    checkpoint_path=data/DFoT_RE10K.ckpt \
    ngpus=1 \
    image_size=128 \
    ctxt_min=5 \
    ctxt_max=15 \
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
    model.encoder.use_depth_mask=true \
    model.encoder.encoder_ckpt=data/re10k.ckpt \
    model.encoder.freeze_depth_predictor=false \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.use_vggt_alignment=true \
    model.encoder.backbone.use_repa=true \
    model.encoder.backbone.input_size='[128, 128]' \
    alignment.latents_info=-1 \
    ctxt_losses_factor=0.9 \
    repa_encoder_resolution=512 \
    model_type=uvit_pose \
    name=spoc_base \
    wandb=online \
    wandb.entity=your_wandb_entity \
    clean_target=false \
    use_identity=true \
    intermediate=true \
    load_optimizer=false \
    load_enc=true \
    lock_enc_steps=5 \
    use_depth_smoothness=true \
    adjacent_angle=0.785 \
    adjacent_distance=1.0 \
    num_intermediate=15

