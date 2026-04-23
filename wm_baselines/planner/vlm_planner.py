from typing import Any, Dict, Union, List
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from omegaconf import DictConfig
from copy import deepcopy
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.utils.common_utils import with_timing
from wm_baselines.agent.vlm.vlm import VLM, GeminiVLM

class VLMPlanner(BasePlanner):
    """VLM-based planner that uses vision-language model for goal selection."""

    def __init__(
            self, 
            vlm: Union[VLM, GeminiVLM], 
            prompt: str, 
            action_horizon: int, 
            rotation_step: float,
            move_step: float,
            **kwargs
        ):
        super().__init__(**kwargs)
        self.vlm = vlm
        self.prompt = prompt
        self.action_horizon = action_horizon
        self.action_history: List[str] = []
        self.position_history: List[np.ndarray] = []
        self.rotation_history: List[float] = []
        self.rotation_step = rotation_step
        self.move_step = move_step
        self.actions: List[str] = []
        self._metrics = {
            "planning_time": 0.0,
            "vlm_input_tokens": 0,
            "vlm_output_tokens": 0,
        }
        self.image = None
        self.text_prompt = None

    def reset(self):
        """Reset the planner state if needed."""
        self.current_asset = None
        self._metrics = {
            "planning_time": 0.0,
            "vlm_input_tokens": 0,
            "vlm_output_tokens": 0,
        }
        self.step = -1
        self.action_history = []
        self.position_history = []
        self.rotation_history = []
        self.actions = []
        self.image = None
        self.text_prompt = None

    def get_next_action(self, current_asset: Dict[str, Any], current_step: int) -> Dict[str, Any]:
        """Get the next action based on current assets and step count."""
        if current_asset is not None and current_step > self.step:
            self.set_state(current_asset)
            self.step += 1
            assert self.step == current_step, f"step mismatch: {self.step} vs {current_step}"
        
        if len(self.actions) > 0:
            trace_asset = {
                "rgb": self.current_asset["rgb"],
                "vlm_response": None,
                "action_history": self.action_history
            }
        else:
            params = {
                "object_name": self.current_asset["object_name"],
                "action_horizon": self.action_horizon,
                "rotation_step": self.rotation_step,
                "move_step": self.move_step,
            }
            if "{action_history}" in self.prompt:
                params["action_history"] = ", ".join(self.action_history)
            if "{position_history}" in self.prompt:
                pos_hist_str = []
                for pos in self.position_history:
                    pos_hist_str.append(f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")
                params["position_history"] = ", ".join(pos_hist_str)
            if "{rotation_history}" in self.prompt:
                rot_hist_str = []
                for yaw in self.rotation_history:
                    rot_hist_str.append(f"{yaw:.1f}")
                params["rotation_history"] = ", ".join(rot_hist_str)
            text_prompt = self.prompt.format(**params)
            image = self.current_asset["rgb"]
            self.text_prompt = text_prompt
            self.image = image
            response, exe_time = self.plan()
            self._metrics["planning_time"] += exe_time
            self.actions = response['parsed']
            self.current_asset["vlm_response"] = response
            self._metrics["vlm_input_tokens"] += response.get("num_input_tokens", 0)
            self._metrics["vlm_output_tokens"] += response.get("num_output_tokens", 0)
            trace_asset = {
                "rgb": self.current_asset["rgb"],
                "vlm_response": self.current_asset["vlm_response"],
                "action_history": self.action_history
            }
        if len(self.actions) == 0:
            action = "turn_right"
        else:
            action = self.actions.pop(0)
        self.action_history.append(action)
        self.position_history.append(self.current_asset["position"])
        self.rotation_history.append(self.current_asset["yaw"])
        action_dict = {"action_name": action, "args": {}}

        return action_dict, trace_asset
    
    @with_timing
    def plan(self) -> Dict[str, Any]:
        image = self.image
        text_prompt = self.text_prompt
        response = self.vlm.prompt_predict_actions(image=image, text_prompt=text_prompt)
        return response
