import sys
import os
import random
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import imageio
from copy import deepcopy
from pathlib import Path
import math
import json
import time
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
import jsonlines
from scipy.spatial.transform import Rotation as R
import hydra
from hydra import initialize_config_dir, compose

ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

# Set environment variables
os.environ["OBJAVERSE_DATA_DIR"] = "/home/ubuntu/jhu-scai-lab/yyin34/spoc/data"
os.environ["OBJAVERSE_HOUSES_DIR"] = "/home/ubuntu/jhu-scai-lab/yyin34/spoc/data/houses_2023_07_28"

from environment.stretch_controller import StretchController
from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from belief_baselines.workspace.base_workspace import BaseWorkspace
from belief_baselines.env_interface.base_env_interface import BaseEnvInterface
from belief_baselines.env_interface.spoc_env_interface import SpocEnvInterface
from belief_baselines.task_manager.base_task_manager import BaseTaskManager
from belief_baselines.task_manager.spoc_obj_searching_task_manager import SpocObjSearchingTaskManager


class SpocEnvTestingWorkspace(BaseWorkspace):
    def __init__(self, config: DictConfig):
        # instantiate stretch controller
        stretch_controller = StretchController(**STRETCH_ENV_ARGS)
        embodied_config = config.embodied_task
        task_manager: BaseTaskManager = SpocObjSearchingTaskManager(config=embodied_config, stretch_controller=stretch_controller)
        env_interface: BaseEnvInterface = SpocEnvInterface(config=embodied_config, task_manager=task_manager)
        super().__init__(config, env_interface, task_manager)

    def run(self):
        save_path = self.embodied_config.trajectory.save_path
        print(f"Loaded {len(self.task_manager.episodes)} episodes from {self.task_manager.episode_root}")
        print(f"Max steps per episode: {self.task_manager.max_steps}")
        ep_save_path = None

        self.env_interface.reset()
        # dummy target pose as 4x4 np array
        target_pose = np.array([[1, 0, 0, 1.20],
                                [0, 1, 0, 0.90],
                                [0, 0, 1, 10.50],
                                [0, 0, 0, 1]], dtype=np.float32)

        self.env_interface.nav_to(target_pose)

        # save using BaseWorkspace.save()
        result_paths = self.save(save_path)
        print(f"Saved results to: {result_paths}")


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg: DictConfig):
    workspace = SpocEnvTestingWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()