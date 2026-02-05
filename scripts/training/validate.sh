source /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/miniconda3/etc/profile.d/conda.sh

conda activate dfm-pixel

nvidia-smi

export MASTER_PORT=$((12000 + RANDOM % 1000))

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export CUDA_VISIBLE_DEVICES=4

python /home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/DFM/experiment_scripts/temporal_inference_pixel_epi.py \
    dataset=spoc_seq \
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
    semantic_config=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/DFM/configurations/semantic/onehot.yaml \
    checkpoint_path=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/DFM/outputs/weights/uvit_mvsplat_repa_128_seq_vggt_refine_semantic/model-28.pt \
    results_folder=/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/DFM/outputs/inference/spoc/dfm/uvit_mvsplat_repa_128_seq_vggt_refine_semantic_scene \

conda deactivate