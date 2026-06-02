<h1 align="center"><a href="https://3d-belief.github.io">3D-Belief: Embodied Belief Inference via Generative 3D World Modeling</a></h1>

<div align="center">

<a href="https://3d-belief.github.io" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/Project_Page-blue" alt="Project Page"></a>
<a href="https://arxiv.org/abs/2605.11367" target="_blank" rel="noopener noreferrer"><img src="https://img.shields.io/badge/arXiv-2605.11367-b31b1b.svg" alt="arXiv"></a>
<a href="https://huggingface.co/datasets/SCAI-JHU/3d-belief"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Data-yellow" alt="HuggingFace"></a>

</div>

![3D-Belief Overview](assets/figure_intro.png)

## 3D-Belief

We propose **3D-Belief**, a generative 3D world model that predicts unseen regions in an explicit, actionable 3D representation from partial observations and updates this belief online as new observations arrive. It enables embodied agents to reason about the 3D world under partial observability and make sequential decisions based on up-to-date beliefs.

- **Multi-hypothesis Belief Sampling**: generates diverse 3D scene completions from partial observations, explicitly representing uncertainty over unobserved regions so the agent can plan against multiple possible world states.
- **Sequential Belief Updating**: refines the 3D belief online at each time step as new observations arrive, ensuring the agent always acts on the most current and consistent world representation.
- **Spatially Consistent Scene Memory**: maintains a coherent 3D memory that preserves previously observed regions accurately while integrating new information, avoiding drift or contradiction across time.
- **Semantically Informed Future Prediction**: leverages semantic queries to guide prediction in unobserved regions, enabling goal-directed imagination about where relevant objects are likely to be found.

## 🎉 News

- **[2026-04]** Code, pretrained checkpoints, and processed data released on [HuggingFace](https://huggingface.co/datasets/SCAI-JHU/3d-belief).
- **[2026-04]** 3D-CORE benchmark released — covering object completion, room completion, and object permanence tasks.
- **[2026-04]** Project website and paper live at [3d-belief.github.io](https://3d-belief.github.io).

## Quick Links

- [Installation](#installation)
  - [Environment Setup](#environment-setup)
  - [Third-Party Packages and Assets](#third-party-packages-and-assets)
- [Data & Checkpoints](#data--checkpoints)
- [Inference](#inference)
- [Vision Evaluation](#vision-evaluation)
- [Embodied Evaluation](#embodied-evaluation)
  - [Object Navigation (AI2-THOR)](#object-navigation-ai2-thor)
  - [3D Contextual Reasoning (3D-CORE)](#3d-contextual-reasoning-3d-core)
- [Repository Structure](#repository-structure)
- [Citation](#citation)

## Installation

### Environment Setup

```bash
git clone https://github.com/3D-Belief/3d-belief
cd 3d-belief
# It takes ~ 5 mins to pull all submodules and their dependencies.
git submodule update --init --recursive
conda create -n 3d-belief python=3.10 pip -y
conda activate 3d-belief
conda install -c conda-forge ninja gcc_linux-64=9 gxx_linux-64=9 swig -y
# Install the version that matches your CUDA version. CUDA 12.1 is used here as an example
conda install -c nvidia cuda=12.1 -y

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Install the version that matches your CUDA version. CUDA 12.1 is used here as an example
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c conda-forge vulkan-tools -y
pip install --no-build-isolation -r requirements.txt
pip install -e .
```

### Third-Party Packages and Assets

```bash
cd third_party/dfot
pip install -r requirements.txt
cd ../spoc
pip install --no-build-isolation -e .
pip install "setuptools==65.5.0" "wheel<0.40"
# Pin the torch stack so spoc's unpinned xformers/torch do not replace the cu121 build
# from the main install (xformers is disabled at runtime via XFORMERS_DISABLED=1).
printf 'torch==2.5.1\ntorchvision==0.20.1\ntorchaudio==2.5.1\ntriton==3.1.0\nxformers==0.0.28.post3\n' > /tmp/3db_torch_constraints.txt
pip install -r requirements.txt -c /tmp/3db_torch_constraints.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246
# Download and unzip the assets may take a while
python -m objathor.dataset.download_annotations --version 2023_07_28 --path ../../data
python -m objathor.dataset.download_assets --version 2023_07_28 --path ../../data
python -m scripts.download_objaverse_houses --save_dir ../../data --subset val
cd ../../
```

## Data & Checkpoints

Log in to your HuggingFace account from the project root:

```bash
hf auth login
```

Download all pretrained checkpoints:

```bash
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "checkpoints/**"
```

Model inference additionally requires the DINOv3 ViT-B/16 backbone
weights at `checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth`. These weights must be obtained separately from Meta's gated DINOv3 release (request access / accept the license at the [DINOv3 GitHub](https://github.com/facebookresearch/dinov3)), then place the
`dinov3_vitb16_pretrain_lvd1689m.pth` file at `checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth`.

Download and set up evaluation data:

### SPOC (AI2-THOR)

```bash
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "data/3d-core.zip" "data/spoc_trajectories_val.zip"
# Unzipping may take several minutes
unzip ./data/3d-core.zip -d ./data/ && rm data/3d-core.zip
unzip ./data/spoc_trajectories_val.zip -d ./data/ && rm data/spoc_trajectories_val.zip
# Expose the unzipped validation trajectories as the `test` split with a symlink for model inference
mkdir -p data/spoc
ln -sfn "$PWD/data/spoc_trajectories_val" data/spoc/test
```

### RealEstate10K

```bash
# Download the per-scene video frames (split zip, ~396 GB; needs ~0.8 TB free for
#    the parts + the reassembled zip) and the camera-pose file:
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "data/re10k/test_parts/*"
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "data/re10k/test.mat"
# Reassemble the single zip from its parts, then extract the per-scene .npz videos:
cat data/re10k/test_parts/re10k_test.zip.part_* > data/re10k/re10k_test.zip
unzip data/re10k/re10k_test.zip -d data/re10k/
# Reclaim space once extraction succeeds:
rm -r data/re10k/test_parts data/re10k/re10k_test.zip
```

## Inference

Two scripts (SPOC and RealEstate10K) run temporal inference. The following commands sample 2 episodes 
from the test datasets and run prediction at an image resolution of 256. Additional command-line overrides include: 
INFERENCE_SAMPLE_FROM_DATASET=false samples episodes from a predefined pool in 
splat_belief/config/inference/temporal_indices.py; INFERENCE_SAVE_SCENE=true saves 
Gaussian scenes as .ply, but will make inference slower.

```bash
# SPOC @ 256
CUDA_VISIBLE_DEVICES=0 IMAGE_SIZE=256 INFERENCE_NUM_SAMPLES=2 \
CHECKPOINT_PATH=checkpoints/3d_belief_spoc_256.pt \
RESULTS_FOLDER=outputs/inference/spoc_256 \
bash scripts/inference/validate.sh

# RealEstate10K @ 256
CUDA_VISIBLE_DEVICES=0 IMAGE_SIZE=256 INFERENCE_NUM_SAMPLES=2 \
CHECKPOINT_PATH=checkpoints/3d_belief_re10k_256.pt \
RESULTS_FOLDER=outputs/inference/re10k_256 \
bash scripts/inference/validate_re10k.sh
```

## Vision Evaluation

We report video-prediction metrics, PSNR/SSIM/LPIPS for scene memory and FVD/FID for scene imagination, on two benchmarks: the AI2-THOR (SPOC) trajectories and RealEstate10K (RE10K). The SPOC evaluation compares 3D-Belief against NWM and DFoT, while the RE10K evaluation compares against DFoT and Gen3C. Gen3C runs in its own conda environment; see [GEN3C.md](GEN3C.md) for the submodule and environment setup before running the RE10K comparison with Gen3C.

### SPOC (AI2-THOR)

To compute metrics for 3D-Belief, NWM, and DFoT on the AI2-THOR (SPOC) trajectories, run:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT="$PWD/outputs/vision_metrics/spoc" \
bash scripts/vision_metrics/run_vision_comparison.sh
```

### RealEstate10K

To compute metrics for 3D-Belief and DFoT on RealEstate10K, first set up the RE10K test data as
described in [Data & Checkpoints](#realestate10k-for-re10k-inference--vision-evaluation) (this
populates `data/re10k`, the default `DATASET_ROOT`), then run:

```bash
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_ROOT="$PWD/outputs/vision_metrics/re10k" \
bash scripts/vision_metrics/run_vision_comparison_re10k.sh
```

(Pass `DATASET_ROOT=/path/to/re10k` if your RE10K data lives elsewhere.)

To include Gen3C as a baseline (after completing the setup in [GEN3C.md](GEN3C.md)), activate its environment and add `gen3c` to `MODELS`:

```bash
# Activate the cosmos-predict1 environment to get GEN3C_PYTHON / GEN3C_CUDA_HOME
conda activate cosmos-predict1
GEN3C_PYTHON=$(which python)
GEN3C_CUDA_HOME=$CONDA_PREFIX
conda deactivate

CUDA_VISIBLE_DEVICES=0 \
DATASET_ROOT=/path/to/re10k \
OUTPUT_ROOT="$PWD/outputs/vision_metrics/re10k_with_gen3c" \
MODELS="3d_belief,dfot,gen3c" \
GEN3C_PYTHON="${GEN3C_PYTHON}" \
GEN3C_CUDA_HOME="${GEN3C_CUDA_HOME}" \
bash scripts/vision_metrics/run_vision_comparison_re10k.sh
```

Results are written to `OUTPUT_ROOT` (default `outputs/vision_metrics/<run-name>/`). `metrics/summary.json` and `metrics/summary.csv` show the aggregated observed and imagined scores.

## Embodied Evaluation

To evaluate on 3D-CORE and the short SPOC object-navigation episodes, please follow the instructions below. See [Example Evaluation Results](RESULTS.md) for a reference of the results produced.

### Object Navigation (AI2-THOR)

For VLM-based models, export your API keys:

```bash
export OPENAI_API_KEY=your_openai_api_key
export GEMINI_API_KEY=your_gemini_api_key
```

Run evaluation for a selected model:

```bash
bash scripts/rollouts/object_searching.sh 3d_belief_semantic_goal_selector
```

Available model keys: `gpt_vlm_agent`, `gemini_vlm_agent`, `qwen3_vlm_agent`, `vggt_frontier`, `vggt_gpt_vlm_goal_selector`, `dfot_vggt_gpt_vlm_goal_selector`

Evaluate predicted trajectories:

```bash
python scripts/calculate_metrics/obj_searching_metrics.py <path_to_predicted_trajectories>
```

### 3D Contextual Reasoning (3D-CORE)

3D-CORE includes three tasks:
- **Object Completion** (`obj_comp_*`)
- **Room Completion** (`room_comp_*`)
- **Object Permanence** (`obj_perm_*`)

We use Gemini-2.5-Flash for evaluation. Export your key:

```bash
export GEMINI_API_KEY=your_gemini_api_key
```

Run one task/model pair:

```bash
bash scripts/rollouts/reasoning.sh obj_comp_3d_belief
```

Available agent keys:
- `obj_comp_3d_belief`, `room_comp_3d_belief`, `obj_perm_3d_belief`
- `obj_comp_dfot_vggt`, `room_comp_dfot_vggt`, `obj_perm_dfot_vggt`

Evaluate each task:

```bash
python scripts/calculate_metrics/obj_comp_metrics.py <path_to_predicted_trajectories>
python scripts/calculate_metrics/room_comp_metrics.py <path_to_predicted_trajectories>
python scripts/calculate_metrics/obj_perm_metrics.py <path_to_predicted_trajectories>
```

## Repository Structure

```
3d-belief/
├── splat_belief/          # 3D-Belief model
│   ├── diffusion/         # Diffusion model wrapper
│   ├── splat/             # 3D Gaussian Splat scene representation
│   ├── embodied/          # Embodied related tools
│   ├── config/            # Model configs
│   └── data_io/           # Data loading utilities
├── wm_baselines/          # Baseline world model agents
│   ├── agent/             # Agent implementations
│   ├── world_model/       # World model wrappers
│   ├── planner/           # Motion planning modules
│   ├── task_manager/      # Task management
│   ├── workspace/         # Evaluation entry points
│   └── config/            # Baseline configs including paths.yaml
├── scripts/
│   ├── training/          # Model training / finetuning scripts
│   ├── inference/         # 3D-Belief inference scripts
│   ├── vision_metrics/    # Vision evaluation scripts
│   ├── rollouts/          # Embodied rollouts scripts
│   └── calculate_metrics/ # Per-task embodied evaluation scripts
└── third_party/           # Submodules
```

## Citation

If you find 3D-Belief useful in your research, please consider giving us a star and citing our paper:

```bibtex
@article{yin2026_3dbelief,
  title   = {3D-Belief: Embodied Belief Inference via Generative 3D World Modeling},
  author  = {Yin, Yifan and Wen, Zehao and Ye, Suyu and Chen, Jieneng and Zheng, Zehan and Dai, Nanru and Shi, Haojun and Huang, Aydan and Zhang, Zheyuan and Yuille, Alan and Xie, Jianwen and Tewari, Ayush and Shu, Tianmin},
  journal = {arXiv preprint arXiv:2605.11367},
  year    = {2026},
  doi     = {10.48550/arXiv.2605.11367},
  url     = {https://arxiv.org/abs/2605.11367}
}
```
