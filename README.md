# 3D Belief

We propose 3D-Belief, a generative 3D world model that builds and updates 3D scene beliefs online to boost embodied agent performance in reasoning and planning tasks.

## Usage

### Environment Setup 
Conda

```bash
git clone https://github.com/3D-Belief/3d-belief
cd 3d-belief
conda create -n 3d-belief python=3.10 -y 
conda activate 3d-belief
conda install -c conda-forge ninja gcc_linux-64=9 gxx_linux-64=9
conda install -c conda-forge moviepy
conda install -c nvidia cuda=12.1

export PATH=$CONDA_PREFIX/bin:$PATH
export CUDA_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
pip install -r requirements.txt
pip install -e .
```

Venv

```bash
git clone https://github.com/3D-Belief/3d-belief
cd 3d-belief
# (optional) Install uv if have not
curl -Ls https://astral.sh/uv/install.sh | sh
# Create a venv
uv init --python 3.10 3db_venv
cd 3db_venv
uv venv
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
uv pip install moviepy
uv pip install fvcore iopath ninja

export CUDA_HOME=/usr/lib/nvidia-cuda-toolkit
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

uv pip install --pre torch --index-url https://download.pytorch.org/whl/cu126
cd ../
export TORCH_CUDA_ARCH_LIST="8.6;9.0"
uv pip install --no-build-isolation -r requirements.txt
uv pip install -e .
```
