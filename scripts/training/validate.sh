conda activate 3d-belief

nvidia-smi

export MASTER_PORT=$((12000 + RANDOM % 1000))

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=5
export TORCH_CUDA_ARCH_LIST="8.6;9.0"

python splat_belief/experiment/temporal_inference.py \
    dataset=procthor \
    dataset.root_dir=datasets/spoc \
    batch_size=1 \
    num_target=1 \
    num_context=1 \
    stage=unit \
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
    model.encoder.use_segmentation=true \
    model/encoder/backbone=u_vit3d_pose \
    model.encoder.backbone.input_size='[128, 128]' \
    model_type=uvit_pose \
    semantic_mode=embed \
    semantic_viz=query \
    temperature=0.85 \
    sampling_steps=50 \
    name=mvsplat_inference_seg \
    image_size=128 \
    inference_sample_from_dataset=true \
    inference_num_samples=5 \
    inference_min_frames=5 \
    inference_max_frames=25 \
    adjacent_angle=0.523 \
    adjacent_distance=1.0 \
    clean_target=false \
    use_history=false \
    inference_save_scene=true \
    semantic_config=splat_belief/config/semantic/onehot.yaml \
    checkpoint_path=checkpoints/3d_belief_spoc.pt \
    results_folder=outputs/inference/procthor_seg \

conda deactivate