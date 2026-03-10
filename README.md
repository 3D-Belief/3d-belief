# 3D Belief

We propose 3D-Belief, a generative 3D world model that builds and updates 3D scene beliefs online to boost embodied agent performance in reasoning and planning tasks.

## Usage

### Environment Setup 

```bash
git clone https://github.com/3D-Belief/3d-belief
cd 3d-belief
conda create -n 3d-belief python=3.10 -y 
conda activate 3d-belief
conda install -c conda-forge ninja gcc_linux-64=9 gxx_linux-64=9
conda install -c conda-forge moviepy swig
conda install -c nvidia cuda=12.1

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
pip install --no-build-isolation -r requirements.txt
pip install -e .
```

### Install Third-Party Packages

```bash
cd third_party/spoc
pip install --no-build-isolation -e ./src/clip
pip install -r requirements.txt
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

