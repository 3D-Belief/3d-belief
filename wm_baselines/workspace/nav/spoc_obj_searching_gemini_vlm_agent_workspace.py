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

class SpocObjSearchingGeminiVLMAgentWorkspace(BaseWorkspace):
    def __init__(self, config: DictConfig):
        # instantiate stretch controller
        stretch_controller = StretchController(**STRETCH_ENV_ARGS)
        super().__init__(config, stretch_controller=stretch_controller)

    def run(self):
        save_path = self.embodied_config.trajectory.save_path
        timeout = self.embodied_config.timeout
        print(f"Loaded {len(self.task_manager.episodes)} episodes from {self.task_manager.episode_root}")
        print(f"Max steps per episode: {self.task_manager.max_steps}")
        ep_save_path = None

        for ep_idx, ep in enumerate(self.task_manager.episodes):
            try:
                print(f"Starting episode {ep_idx}/{len(self.task_manager.episodes)}: {self.task_manager.current_ep_name}")
                self.env_interface.reset()
                self.agent.reset()
                current_ep_name = self.task_manager.current_ep_name
                ep_save_path = os.path.join(save_path, current_ep_name) if save_path else None
                if ep_save_path:
                    os.makedirs(ep_save_path, exist_ok=True)
                step = 0
                done = False
                while (step < timeout) and (not done):
                    step += 1
                    self.agent.observe()
                    action = self.agent.plan()
                    if action["action_name"] == "end":
                        print("Episode ended")
                        done = True
                        break
                    self.agent.act(action)
                    done = self.task_manager.is_done()
                    print(f"Step {step}/{timeout}, Action: {action['action_name']}, Done: {done}")

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
    workspace = SpocObjSearchingGeminiVLMAgentWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()