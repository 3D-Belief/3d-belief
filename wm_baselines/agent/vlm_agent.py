from typing import Any, Dict, List, Optional, Union
import numpy as np
import os
import torch
import json
from torch import Tensor
from pathlib import Path
from PIL import Image
#import open3d as o3d
from wm_baselines.env_interface.base_env_interface import BaseEnvInterface
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.agent.base_agent import BaseAgent
from wm_baselines.agent.perception.camera import Camera
from wm_baselines.utils.vision_utils import to_pil

class VLMAgent(BaseAgent):
    """VLM agent that uses vision-language model for object searching."""

    def __init__(self,
                 env_interface: BaseEnvInterface, 
                 world_model: BaseWorldModel, 
                 planner: BasePlanner,
                 camera: Camera,
                 assets_saver: Dict[str, Any],
                 **kwargs
        ):
        super().__init__(env_interface, world_model, planner, camera=camera, assets_saver=assets_saver)
        self.current_ep_name = None
        self.current_object_name = None
    
    def reset(self):
        """Reset the agent state if needed. Subclasses can override."""
        self.world_model.reset()
        self.planner.reset()
        self.current_observation = None
        self.current_assets = None
        self.step = -1
        self.current_ep_name = self.env_interface.task_manager.current_ep_name
        self.current_object_name = self.env_interface.task_manager.current_target_obj
        self._setup_assets_save()

    def plan(self) -> Dict[str, Any]:
        """Plan the next action. """
        self.current_assets["object_name"] = self.current_object_name
        action, trace_asset = self.planner.get_next_action(self.current_assets, self.step)
        self._step_assets_save(trace_asset)
        return action

    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Given an action, invoke env_interface and return the resulting observation dict."""
        action_fn = getattr(self.env_interface, action["action_name"])
        return action_fn(**action["args"])
    
    def _step_assets_save(self, trace_asset: Dict[str, Any]) -> None:
        """Return the current step's assets for saving/logging."""
        if self.save_assets and self.assets_save_path and self.assets_save_path_ep:
            # Save the specified assets to the designated path
            for key in self.assets_save_keys:
                if key=='rgb':
                    if trace_asset.get('rgb', None) is None: continue
                    rgb = trace_asset['rgb']
                    rgb = Image.fromarray(rgb)
                    rgb.save(self.assets_save_path_ep / key / f"rgb_{self.step}.png")
                elif key=='vlm_response':
                    vlm_response = trace_asset.get('vlm_response', None)
                    if vlm_response is None: continue
                    # save in json format
                    with open(self.assets_save_path_ep / key / f"vlm_response_{self.step}.json", 'w') as f:
                        json.dump(vlm_response, f, indent=4)
                elif key=='action_history':
                    action_history = trace_asset.get('action_history', None)
                    if action_history is None: continue
                    # save in txt format
                    with open(self.assets_save_path_ep / key / f"action_history_{self.step}.txt", 'w') as f:
                        f.write("\n".join(action_history))
