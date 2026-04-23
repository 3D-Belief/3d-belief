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
from hydra.utils import instantiate

ROOT_DIR = str(Path(__file__).parent.parent.parent.parent)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

from environment.stretch_controller import StretchController
from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from wm_baselines.workspace.base_workspace import BaseWorkspace
from wm_baselines.env_interface.base_env_interface import BaseEnvInterface
from wm_baselines.task_manager.base_task_manager import BaseTaskManager
from wm_baselines.agent.base_agent import BaseAgent
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.agent.vlm.vlm import VLM

class SpocObjCompletion3DBeliefWorkspace(BaseWorkspace):
    def __init__(self, config: DictConfig):
        # instantiate stretch controller
        stretch_controller = StretchController(**STRETCH_ENV_ARGS)
        super().__init__(config, stretch_controller=stretch_controller)

    def run(self):
        # Initialize VLM
        vlm_model_name = self.embodied_config.vlm_model_name
        vlm = VLM(vlm_model_name)
        self.task_manager.set_vlm(vlm)

        save_path = self.embodied_config.trajectory.save_path
        print(f"Loaded {len(self.task_manager.episodes)} episodes from {self.task_manager.episode_root}")
        self.task_manager.set_camera(self.agent.camera)

        for ep_idx, ep in enumerate(self.task_manager.episodes):
            try:
                self.env_interface.reset()
                self.agent.reset()
                print(f"Starting episode {ep_idx}/{len(self.task_manager.episodes)}: {self.task_manager.current_ep_name}")
                max_steps = self.task_manager.num_steps
                current_ep_name = self.task_manager.current_ep_name
                ep_save_path = os.path.join(save_path, current_ep_name) if save_path else None
                if ep_save_path:
                    os.makedirs(ep_save_path, exist_ok=True)
                step = 0
                done = False
                while (step < max_steps) and (not done):
                    step += 1
                    self.agent.observe()
                    # ## DEBUG
                    # if not self.task_manager.done:
                    #     continue
                    self.agent.imagine()
                    self.agent.calculate_metrics()
                    self.agent.save_step()
                    done = self.task_manager.is_done()
                    print(f"Step {step}/{max_steps}, Done: {done}")

                # save using BaseWorkspace.save()
                result_paths = self.save(ep_save_path)
                print(f"Saved results to: {result_paths}")
            except:
                result_paths = self.save(ep_save_path)
                print(f"Error in episode {self.task_manager.current_ep_name}, skipping to next episode.")
                continue


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg: DictConfig):
    workspace = SpocObjCompletion3DBeliefWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()