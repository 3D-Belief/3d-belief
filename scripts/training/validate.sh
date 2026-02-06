source uv_venv/.venv/bin/activate

nvidia-smi

export MASTER_PORT=$((12000 + RANDOM % 1000))

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

python splat_belief/experiment/temporal_inference.py \
    dataset=spoc_seq \
    dataset.root_dir=/path/to/dataset \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    stage=test \
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
    temperature=0.85 \
    sampling_steps=50 \
    name=mvsplat_inference \
    image_size=128 \
    inference_sample_from_dataset=true \
    inference_num_samples=25 \
    inference_min_frames=5 \
    inference_max_frames=25 \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    clean_target=false \
    use_history=false \
    inference_save_scene=true \
    semantic_config=splat_belief/config/semantic/onehot.yaml \
    checkpoint_path=/path/to/checkpoint \
    results_folder=outputs/inference/spoc_base \

conda deactivate