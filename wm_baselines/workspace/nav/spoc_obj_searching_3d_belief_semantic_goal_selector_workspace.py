import sys
import os
import random
import pathlib
import traceback
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

class SpocObjSearching3DBeliefSemanticGoalSelectorWorkspace(BaseWorkspace):
    def __init__(self, config: DictConfig):
        # instantiate stretch controller
        stretch_controller = StretchController(**STRETCH_ENV_ARGS)
        super().__init__(config, stretch_controller=stretch_controller)

    def run(self):
        save_path = self.embodied_config.trajectory.save_path
        timeout = self.embodied_config.timeout
        self.agent.planner.set_world_model(self.agent.world_model)
        n_eps = len(self.task_manager.episodes)
        print(f"Loaded {n_eps} episodes from {self.task_manager.episode_root}")
        print(f"Max steps per episode: {self.task_manager.max_steps}")
        # Optional env-var slicing for smoke tests, e.g. WM_BASELINES_START_EP=6 WM_BASELINES_END_EP=7
        start_ep = int(os.environ.get("WM_BASELINES_START_EP", "0"))
        end_ep = int(os.environ.get("WM_BASELINES_END_EP", str(n_eps)))
        # When set, surface per-episode exceptions instead of swallowing them.
        raise_on_error = bool(int(os.environ.get("WM_BASELINES_RAISE", "0")))
        ep_save_path = None

        for ep_idx in range(start_ep, min(end_ep, n_eps)):
            try:
                if self.seed is not None:
                    self._set_seed(self.seed + ep_idx)
                self.env_interface.reset(idx=ep_idx)
                self.agent.reset()
                current_ep_name = self.task_manager.current_ep_name
                print(f"Starting episode {ep_idx}/{n_eps}: {current_ep_name}")
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
            except Exception:
                traceback.print_exc()
                if raise_on_error:
                    raise
                try:
                    result_paths = self.save(ep_save_path)
                except Exception:
                    traceback.print_exc()
                print(f"Error in episode {self.task_manager.current_ep_name}, skipping to next episode.")
                continue


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg: DictConfig):
    workspace = SpocObjSearching3DBeliefSemanticGoalSelectorWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()