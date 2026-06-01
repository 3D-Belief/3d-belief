# Gen3C Setup

[Gen3C](https://github.com/nv-tlabs/GEN3C) is an optional baseline for the RealEstate10K vision evaluation. This document covers its setup; the commands for running the RE10K comparison live in the [Vision Evaluation](README.md#vision-evaluation) section of the main README.

1. [Submodule](#submodule)
2. [Environment](#environment)
3. [Checkpoints](#checkpoints)
4. [Environment variables for the RE10K runner](#environment-variables-for-the-re10k-runner)

## Submodule

Gen3C is included as a git submodule under `third_party/gen3c`. Initialize it after cloning:

```bash
git submodule update --init third_party/gen3c
```

## Environment

Gen3C requires its own conda environment (`cosmos-predict1`), separate from the main `3d-belief` environment:

```bash
cd third_party/gen3c

# Create and activate the cosmos-predict1 environment
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1

# Install dependencies
pip install -r requirements.txt

# Patch Transformer engine linking issues in conda environments
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10

# Install Transformer engine
pip install transformer-engine[pytorch]==1.12.0

# Install Apex for inference
git clone https://github.com/NVIDIA/apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir \
    --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" \
    --config-settings "--build-option=--cuda_ext" \
    ./apex

# Install MoGe (depth estimator used for RE10K, which has no simulator depth)
pip install git+https://github.com/microsoft/MoGe.git
pip install huggingface-hub==0.29.2

cd ../..
```

Verify the environment:

```bash
cd third_party/gen3c
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/test_environment.py
cd ../..
```

## Checkpoints

```bash
cd third_party/gen3c
# Log in with a Hugging Face token (Read permission is enough; needed for the gated Cosmos repos)
huggingface-cli login
# Downloads Gen3C-Cosmos-7B + Cosmos-Tokenize1-CV8x8x8-720p + google-t5/t5-11b (+ guardrail)
CUDA_HOME=$CONDA_PREFIX PYTHONPATH=$(pwd) python scripts/download_gen3c_checkpoints.py --checkpoint_dir checkpoints
cd ../..
```

This populates `third_party/gen3c/checkpoints/` (the default `--gen3c-checkpoint-dir`). The download
is large (the `t5-11b` text encoder alone is tens of GB).

## Environment variables for the RE10K runner

Once setup is complete, run the RE10K comparison with the commands in the main README's [Vision Evaluation](README.md#vision-evaluation) section. The runner (`scripts/vision_metrics/run_vision_comparison_re10k.sh`) accepts the following variables (defaults shown).

### RE10K (3D-Belief + DFoT)

| Variable | Default | Description |
|---|---|---|
| `DATASET_ROOT` | `data/re10k` | Path to the RealEstate10K dataset |
| `OUTPUT_ROOT` | `outputs/vision_metrics/<run-name>/` | Directory for predictions and metrics |
| `MODELS` | `3d_belief,dfot` | Comma-separated models to run (`3d_belief`, `dfot`, `gen3c`) |
| `IMAGE_SIZE` | `256` | Inference resolution (must match the checkpoint) |
| `NUM_EPISODES` | `200` | Number of episodes to evaluate |
| `BELIEF_CHECKPOINT` | `checkpoints/3d_belief_re10k_256.pt` | 3D-Belief RE10K checkpoint |
| `DFOT_CHECKPOINT` | `checkpoints/DFoT_RE10K.ckpt` | DFoT RE10K checkpoint |
| `BELIEF_CONFIG_PROFILE` | `re10k_256_from_128` | Architecture profile (`re10k_128_vggt` or `re10k_256_from_128`) |

### Gen3C (when `gen3c` is included in `MODELS`)

| Variable | Default | Description |
|---|---|---|
| `GEN3C_PYTHON` | `python` | Python interpreter from the `cosmos-predict1` env |
| `GEN3C_CUDA_HOME` | unset | `CUDA_HOME` for the `cosmos-predict1` env |
| `GEN3C_REPO` | `third_party/gen3c` | Path to the Gen3C repo |
| `GEN3C_CHECKPOINT_DIR` | `<GEN3C_REPO>/checkpoints` | Directory containing Gen3C model weights |
| `GEN3C_MISSING_DEPTH_POLICY` | `moge` | Depth estimation policy for RE10K (no simulator depth) |
| `GEN3C_HEIGHT` / `GEN3C_WIDTH` | `704` / `1280` | Gen3C output resolution |
| `GEN3C_NUM_STEPS` | `35` | Diffusion sampling steps |
| `GEN3C_SEED` | `1` | Random seed for reproducibility |
| `GEN3C_OFFLOAD` | `0` | Set to `1` to offload model to CPU between calls (saves VRAM) |

> **Note**: RE10K has no simulator depth, so Gen3C uses MoGe (`--missing-depth-policy moge`) to estimate depth on the fly. This requires MoGe to be installed in the `cosmos-predict1` environment (see [Environment](#environment) above).
