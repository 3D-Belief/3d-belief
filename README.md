# 3D Belief

We propose 3D-Belief, a generative 3D world model that builds and updates 3D scene beliefs online to boost embodied agent performance in reasoning and planning tasks.

## Usage

### Environment Setup 

```bash
git clone https://github.com/3D-Belief/3d-belief
cd 3d-belief
git submodule update --init --recursive
conda create -n 3d-belief python=3.10 -y 
conda activate 3d-belief
conda install -c conda-forge ninja gcc_linux-64=9 gxx_linux-64=9 moviepy swig
conda install -c nvidia cuda=12.1 # Use the one matches your cuda version

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda config --add channels conda-forge; conda install vulkan-tools
pip install --no-build-isolation -r requirements.txt
pip install -e .
```

### Install Third-Party Packages

```bash
cd third_party/dfot
pip install -r requirements.txt
cd ../spoc
pip install --no-build-isolation -e .
pip install -r requirements.txt
pip install --extra-index-url https://ai2thor-pypi.allenai.org ai2thor==0+5d0ab8ab8760eb584c5ae659c2b2b951cab23246
python -m scripts.download_training_data --save_dir ../../data --types all
python -m objathor.dataset.download_annotations --version 2023_07_28 --path ../../data
python -m objathor.dataset.download_assets --version 2023_07_28 --path ../../data
python -m scripts.download_objaverse_houses --save_dir ../../data --subset val

```


### Download Data and Pretrained Checkpoints
From the project root directory, open a terminal and log in to your Hugging Face account.

```bash
hf auth login
```
Enter your password. You can now download the assets.

The following commands download all the checkpoints under a created checkpoints/ directory.

```bash
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "checkpoints/**" 
```

Then download and set up the assets under data/ directory.

```bash
hf download SCAI-JHU/3d-belief --repo-type dataset --local-dir ./ --include "data/**"
unzip ./data/spoc_trajectories_val.zip -d ./data/ && rm data/spoc_trajectories_val.zip
```

### Benchmark Path Planning
To benchmark the performance of 3d-Belief and other baseline models on the object searching task, you need to set up the paths first at `wm_baselines/config/paths.yaml`.

Then, if you want to run VLM-based models, export your keys:
```bash
export OPENAI_API_KEY=...
export GOOGLE_API_KEY=...
```

Run one model (`run_obj_searching_single`):
```bash
bash scripts/rollouts/object_searching.sh 3d_belief_semantic_goal_selector
```
You can replace the model key with others such as:
`gpt_vlm_agent`, `gemini_vlm_agent`, `qwen3_vlm_agent`, `vggt_frontier`,
`vggt_gpt_vlm_goal_selector`, `dfot_vggt_gpt_vlm_goal_selector`

To run all the models sequentially:
```bash
bash scripts/rollouts/object_searching_single.sh
```
Since all the models run on a single GPU, this script may take a quite a while.

Then, after a model rollout is done, you can evaluate the model's performance at:
```bash
python scripts/calculate_metrics/obj_searching_metrics.py <path_to_predicted_trajectories>
```

### Benchmark 3D-CORE

3D-CORE includes three tasks:
- object completion (`obj_comp_*`)
- room completion (`room_comp_*`)
- object permanence (`obj_perm_*`)

Run one task/model pair with:
```bash
bash scripts/rollouts/reasoning.sh obj_comp_3d_belief
```
Available agent keys:
- `obj_comp_3d_belief`, `room_comp_3d_belief`, `obj_perm_3d_belief`
- `obj_comp_dfot_vggt`, `room_comp_dfot_vggt`, `obj_perm_dfot_vggt`

We use Gemini-2.5-Flash in evaluation, so make sure to export your key accordingly:
```bash
export GOOGLE_API_KEY=...
```

Evaluate each reasoning task with:
```bash
python scripts/calculate_metrics/obj_comp_metrics.py <path_to_predicted_trajectories>
python scripts/calculate_metrics/room_comp_metrics.py <path_to_predicted_trajectories>
python scripts/calculate_metrics/obj_perm_metrics.py <path_to_predicted_trajectories>
```
