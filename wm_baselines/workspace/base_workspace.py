import os
import json
import jsonlines
import imageio
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
from hydra.utils import instantiate
from wm_baselines.env_interface.base_env_interface import BaseEnvInterface
from wm_baselines.task_manager.base_task_manager import BaseTaskManager
from wm_baselines.agent.base_agent import BaseAgent
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.base_planner import BasePlanner
import numpy as np
import random
import torch


class BaseWorkspace:
    """Base workspace class to manage the environment interface and task manager.

    Responsibilities:
    - hold config reference
    - initialize and hold references to EnvInterface and BaseTaskManager
    - define a run() method to execute the main loop (to be implemented by subclasses)
    """

    def __init__(self, config: DictConfig, **kwargs: Any):
        self.config = config
        self.embodied_config = config.embodied_task
        self.task_manager: BaseTaskManager = instantiate(self.config.task_manager, embodied_config=self.embodied_config, **kwargs)
        self.env_interface: BaseEnvInterface = instantiate(self.config.env_interface, embodied_config=self.embodied_config, task_manager=self.task_manager, **kwargs)
        self.agent: BaseAgent = instantiate(self.config.agent, env_interface=self.env_interface, **kwargs)
        self.seed = config.get("seed", None)
        if self.seed is not None:
          self._set_seed(self.seed)

    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # NOTE: Uncomment if you want to make cudnn deterministic
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False  
        if hasattr(self.env_interface, 'seed'):
          self.env_interface.seed(seed)

    def run(self):
        """Run the main loop of the workspace. Must be implemented by subclass."""
        raise NotImplementedError()

    def save(self, save_path: Optional[str] = None) -> Dict[str, str]:
        """Centralized save.

        - Saves image buffer as a video (if available).
        - Saves string buffers as a jsonl (if available).
        - Saves final log to `final_log.json` using task_manager.get_final_log().

        Returns a dict with paths written.
        """
        results = {}
        sp = save_path or getattr(self.embodied_config, "trajectory", None) and getattr(self.embodied_config.trajectory, "save_path", None)
        if not sp:
            return results

        # images
        img_buf = getattr(self.env_interface, "image_buffer", None)
        str_buf = getattr(self.env_interface, "string_buffer", None)
        if img_buf and "rgb" in img_buf and len(img_buf["rgb"]) > 0:
            video_path = os.path.join(sp, f"episode_{getattr(self.task_manager, 'current_episode_index', 0)}_navigation.mp4")
            try:
                imageio.mimwrite(video_path, img_buf['rgb'], fps=5, quality=8)
                results['video'] = video_path
            except Exception:
                results['video_error'] = "failed to write video"

        if str_buf and len(str_buf) > 0:
            jsonl_path = os.path.join(sp, f"episode_{getattr(self.task_manager, 'current_episode_index', 0)}_navigation.jsonl")
            try:
                with jsonlines.open(jsonl_path, mode='w') as writer:
                    # assume all string buffers are same length
                    length = len(next(iter(str_buf.values()))) if len(str_buf) > 0 else 0
                    for i in range(length):
                        entry = {key: str_buf[key][i] for key in str_buf}
                        writer.write(entry)
                results['jsonl'] = jsonl_path
            except Exception:
                results['jsonl_error'] = "failed to write jsonl"

        # final log
        try:
            world_model_metrics = self.agent.world_model.metrics
            planner_metrics = self.agent.planner.metrics
            planner_metrics.update({
                "rgb": self.agent.current_observation['rgb']
            })
            metrics = {**world_model_metrics, **planner_metrics}
            final_log = self.task_manager.get_final_log(metrics)
            if 'rgb' in final_log:
                del final_log['rgb']
            with open(os.path.join(sp, f"final_log.json"), "w") as f:
                json.dump(final_log, f, indent=4)
            results['final_log'] = os.path.join(sp, f"final_log.json")
        except Exception:
            results['final_log_error'] = "failed to write final log"

        return results