# Training

This guide walks through training 3D-Belief on two datasets:

- **SPOC** (AI2-THOR trajectories) — indoor embodied scenes.
- **RealEstate10K (RE10K)** — real-world video sequences.

Both follow the same two-stage recipe (base training at 128×128, then a fine-tune
at 256×256), plus an optional semantic head fine-tune. The sections below
cover the checkpoints, data setup, and run commands for each.

Prerequisites: the `3d-belief` conda environment and the third-party submodules
described in [README.md](README.md#installation). The training scripts auto-activate
`conda activate 3d-belief`.

---

## 1. Pretrained checkpoints

Several pretrained weights are used as initialization or for auxiliary losses.

| File | Used by | Source |
|---|---|---|
| `checkpoints/DFoT_RE10K.ckpt` | `train.sh`, `train_re10k.sh` (base init) | `SCAI-JHU/3d-belief` (HF), upstream **DFoT** (see [Acknowledgements](#acknowledgements)) |
| `checkpoints/re10k.ckpt` | `train.sh`, `train_re10k.sh` (encoder init) | `SCAI-JHU/3d-belief` (HF), upstream **MVSplat** (see [Acknowledgements](#acknowledgements)) |
| `checkpoints/3d_belief_spoc.pt` | `train_256_from_128.sh`, `finetune_semantic.sh` | `SCAI-JHU/3d-belief` (HF) |
| `checkpoints/3d_belief_re10k.pt` | `train_256_from_128_re10k.sh` (resume) | `SCAI-JHU/3d-belief` (HF) |
| `checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth` | REPA encoder / semantic regression (all stages) | **Meta gated release** ([DINOv3 GitHub](https://github.com/facebookresearch/dinov3)) |

Download the HuggingFace-hosted checkpoints from the repo root:

```bash
hf auth login   # only required the first time
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "checkpoints/**"
```

### DINOv3 backbone (obtain separately from Meta)

`checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth` is **not** part of the HuggingFace,
and it must be obtained from Meta's gated DINOv3 release. As in the main
[README](README.md#data--checkpoints): request access / accept the license at the
[DINOv3 GitHub](https://github.com/facebookresearch/dinov3), download the
**ViT-B/16** (`dinov3_vitb16_pretrain_lvd1689m.pth`) weights, and place the file at:

```
checkpoints/dinov3_vitb16_pretrain_lvd1689m.pth
```

---

## 2. Train on SPOC

### 2.1 Download and extract the SPOC training data

The re-rendered SPOC trajectories live in `data/all_rerendered_root_parts/` on the
HuggingFace dataset as the byte-split parts of a single zip
(`all_rerendered_root.zip.part_aa … part_ai`, ~881 GB total; budget ~1.8 TB free for
the parts plus the reassembled zip). Download the parts, reassemble the zip, and
extract it:

```bash
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ \
    --include "data/all_rerendered_root_parts/*"

# Reassemble the single zip from its parts, then extract the per-scene trajectories:
cat data/all_rerendered_root_parts/all_rerendered_root.zip.part_* > data/all_rerendered_root.zip
unzip data/all_rerendered_root.zip -d data/        # -> data/all_rerendered_root/train/<scene>/

# Reclaim space once extraction succeeds:
rm -r data/all_rerendered_root_parts data/all_rerendered_root.zip
```

This produces the per-scene training folders under `data/all_rerendered_root/train/`.

### 2.2 Lay out `data/spoc/`

`splat_belief/data_io/spoc.py` loads scenes by globbing `data/spoc/<split>/<scene>/`,
where `<split>` is one of `train`, `test`, or `unit`. Wire the extracted training
scenes into the `train` split and the validation set into `test` (symlinks keep disk
shared with the downloads):

```bash
mkdir -p data/spoc
ln -sfn "$PWD/data/all_rerendered_root/train" data/spoc/train
# Validation split (the val zip from the README Data & Checkpoints section):
ln -sfn "$PWD/data/spoc_trajectories_val" data/spoc/test
```

After this `data/spoc/` should look like:

```
data/spoc/
├── train/
│   ├── <scene_0>/
│   └── ...
└── test/
```

### 2.3 Stage 1 — base training at 128×128

Trains the encoder + backbone from the RE10K-pretrained weights at resolution 128.
Defaults to `outputs/training/spoc_base/`.

```bash
# bash scripts/training/train.sh <NGPUS> <WANDB_MODE> <WANDB_ENTITY>
bash scripts/training/train.sh 4 online your-wandb-entity
```

Common overrides (env vars before the call):

- `NGPUS=4` — number of GPUs (also the first positional arg)
- `WANDB=local` / `WANDB=online` — wandb mode (positional arg 2)
- `CUDA_VISIBLE_DEVICES=0,1,2,3` — pin specific devices

The script auto-chooses a unique `MASTER_PORT` so concurrent runs don't collide.

### 2.4 Stage 2 — fine-tune at 256×256

Resumes from the 128-resolution checkpoint and continues at 256. Defaults to
`outputs/training/spoc_base_256_from_128/`, resuming from
`checkpoints/3d_belief_spoc.pt`.

```bash
bash scripts/training/train_256_from_128.sh 4 online your-wandb-entity
```

To resume from your own 128 checkpoint instead:

```bash
CHECKPOINT_PATH=path/to/your/checkpoint \
bash scripts/training/train_256_from_128.sh 4 online your-wandb-entity
```

### 2.5 Semantic head fine-tune (optional)

Fine-tunes the optional semantic prediction head on top of the SPOC base checkpoint. It
enables `use_semantic=true` and a DINOv3-based semantic regression model, and unfreezes
only the semantic head (`finetune_component=semantic_head`), so it converges much faster
than a full pretraining run. `scripts/training/finetune_semantic.sh` →
`outputs/training/spoc_semantic/`, on top of `checkpoints/3d_belief_spoc.pt`:

```bash
bash scripts/training/finetune_semantic.sh 4 online your-wandb-entity
```

Override `DATASET_ROOT` / `CHECKPOINT_PATH` via env vars if your data or weights live
elsewhere.

---

## 3. Train on RealEstate10K (RE10K)

### 3.1 Download the RE10K training data

The RE10K data lives under `data/re10k/` on the HuggingFace dataset:

```
data/re10k/
├── train.mat          # camera poses for the train scenes (~922 MB)
├── test.mat           # camera poses for the test scenes (~100 MB)
├── train_parts/       # re10k_train_part01.zip … partNN.zip  (~3.58 TB total)
└── test_parts/        # re10k_test.zip.part_*  (byte-split single zip, ~396 GB)
```

#### Training scenes

```bash
# Pose file + the split training archives.
#   train_parts is ~3.58 TB; make sure you have room for the archives *and* the
#   extracted .npz scenes (delete each zip after it extracts if space is tight).
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "data/re10k/train.mat"
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "data/re10k/train_parts/*"

# Extract & merge every part archive into data/re10k/  (-> data/re10k/train/<scene>.npz)
for z in data/re10k/train_parts/re10k_train_part*.zip; do
    echo "Extracting $z"
    unzip -n "$z" -d data/re10k/
done

# Reclaim space once extraction succeeds:
rm -r data/re10k/train_parts
```

#### Validation scenes

Set up
`data/re10k/test/` and `data/re10k/test.mat` as well. Follow the RealEstate10K
download steps in the main [README](README.md#realestate10k).


### 3.2 Stage 1 — base training at 128×128

`scripts/training/train_re10k.sh` trains the encoder + backbone at resolution 128,
initialized from `checkpoints/DFoT_RE10K.ckpt` (base) and `checkpoints/re10k.ckpt`
(encoder), with the DINOv3 REPA encoder. Output lands in
`outputs/training/re10k_base/`.

```bash
# bash scripts/training/train_re10k.sh <NGPUS> <WANDB_MODE> <WANDB_ENTITY>
bash scripts/training/train_re10k.sh 4 online your-wandb-entity
```

Overrides (env vars before the call):

- `NGPUS=4` — number of GPUs (also the first positional arg; default 1)
- `WANDB=online|local` and `WANDB_ENTITY=…` — wandb mode / entity (positional args 2/3)
- `CUDA_VISIBLE_DEVICES=0,1,2,3` — pin specific devices

### 3.3 Stage 2 — fine-tune at 256×256

`scripts/training/train_256_from_128_re10k.sh` resumes from a 128-resolution RE10K
checkpoint and continues at resolution 256. It defaults to resuming from
`checkpoints/3d_belief_re10k.pt` and writes to
`outputs/training/re10k_base_256_from_128/`.

```bash
bash scripts/training/train_256_from_128_re10k.sh 4 online your-wandb-entity
```

To resume from your own Stage-1 RE10K run instead of the released checkpoint:

```bash
CHECKPOINT_PATH=path/to/your/checkpoint \
bash scripts/training/train_256_from_128_re10k.sh 4 online your-wandb-entity
```

### 3.4 Semantic head fine-tune (optional)

The RE10K counterpart of [§2.5](#25-semantic-head-fine-tune-optional). RE10K is
real-world video with no GT depth, so this script disables depth supervision and the
depth mask and uses a larger context window.
`scripts/training/finetune_semantic_re10k.sh` → `outputs/training/re10k_semantic/`, on
top of `checkpoints/3d_belief_re10k.pt`:

```bash
bash scripts/training/finetune_semantic_re10k.sh 4 online your-wandb-entity
```

Override `DATASET_ROOT` / `CHECKPOINT_PATH` via env vars if your data or weights live
elsewhere.

---

## 4. Where outputs land

Every script writes to a `results_folder` under `outputs/training/`:

```
outputs/training/
├── spoc_base/                  # train.sh
├── spoc_base_256_from_128/     # train_256_from_128.sh
├── re10k_base/                 # train_re10k.sh
├── re10k_base_256_from_128/    # train_256_from_128_re10k.sh
├── spoc_semantic/              # finetune_semantic.sh
└── re10k_semantic/             # finetune_semantic_re10k.sh
```

Each run directory contains the Hydra config, periodic checkpoints, and wandb local
logs (when `WANDB=local`).

---

## Acknowledgements

Some of the code and the RE10K-pretrained weights we use (`checkpoints/DFoT_RE10K.ckpt`
and `checkpoints/re10k.ckpt`) are from these two awesome repositories:

- [kwsong0113/diffusion-forcing-transformer](https://github.com/kwsong0113/diffusion-forcing-transformer) (`DFoT_RE10K.ckpt`)
- [donydchen/mvsplat](https://github.com/donydchen/mvsplat) (`re10k.ckpt`)

We thank the authors for releasing their code and model weights!
