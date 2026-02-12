from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import json
from torch import Tensor
from pathlib import Path
from PIL import Image
import open3d as o3d
from wm_baselines.env_interface.base_env_interface import BaseEnvInterface
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.agent.base_agent import BaseAgent
from wm_baselines.agent.perception.camera import Camera
from wm_baselines.utils.vision_utils import to_pil, flip_yaw_in_Twc

class ExplorationAgent(BaseAgent):
    """Exploration agent that actively explores the environment."""

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
        self.current_object_name = self.env_interface.task_manager.fix_object_name(self.env_interface.task_manager.current_target_obj)
        self._setup_assets_save()
        self.force_update = False

    def plan(self) -> Dict[str, Any]:
        """Plan the next action. """
        self.current_assets["object_name"] = self.current_object_name
        if len(self.planner.current_trajectory) == 0:
            self.force_update = True
        else:
            self.force_update = False
        obs = self.env_interface.get_observation()
        pose = obs.get("pose")
        if self.world_model.initial_location.get('pose') is not None:
            pose_map = np.linalg.inv(self.world_model.initial_location['pose']) @ pose
            pose_map = self.world_model._project_pose_to_initial_ground(pose_map)
            self.current_assets['pose'] = pose_map
        action, trace_asset = self.planner.get_next_action(self.current_assets, self.step, self.force_update)
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
                if key=='occupancy':
                    occupancy = trace_asset.get('occupancy', None)
                    if occupancy is None: continue
                    path = trace_asset.get('path', None)
                    if "all_goals" in trace_asset:
                        goals = trace_asset['all_goals']
                    else:
                        goals = [trace_asset['goal']]
                    occupancy.save_height_map(self.assets_save_path_ep / key / f"height_map_{self.step}.png", path=path if (path is not None and len(path) > 0) else None)
                    occupancy.save_occupancy_map(self.assets_save_path_ep / key / f"occupancy_map_{self.step}.png", goals=goals if (goals is not None and len(goals) > 0) else None)
                elif key=='pcd':
                    if trace_asset.get('pcd', None) is None: continue
                    o3d.io.write_point_cloud(str(self.assets_save_path_ep / key / f"pcd_{self.step}.ply"), trace_asset['pcd'])
                elif key=='rgb':
                    if trace_asset.get('rgb', None) is None: continue
                    rgb = trace_asset['rgb']
                    rgb = Image.fromarray(rgb)
                    rgb.save(self.assets_save_path_ep / key / f"rgb_{self.step}.png")
                elif key=='goal_images':
                    goal_images = trace_asset.get('goal_images', None)
                    if goal_images is None: continue
                    for i, img in enumerate(goal_images):
                        img = to_pil(img)
                        img.save(self.assets_save_path_ep / key / f"goal_img_{self.step}_{i}.png")
                elif key=='goal_semantics':
                    goal_semantics = trace_asset.get('goal_semantics', None)
                    if goal_semantics is None: continue
                    for i, img in enumerate(goal_semantics):
                        img = to_pil(img)
                        img.save(self.assets_save_path_ep / key / f"goal_semantic_{self.step}_{i}.png")
                elif key=='vlm_scores':
                    vlm_scores = trace_asset.get('vlm_scores', None)
                    if vlm_scores is None: continue
                    with open(self.assets_save_path_ep / key / f"vlm_scores_{self.step}.txt", 'w') as f:
                        for score in vlm_scores:
                            f.write(f"{score}\n")
                elif key=='goal_scores':
                    goal_scores = trace_asset.get('goal_scores', None)
                    if goal_scores is None: continue
                    with open(self.assets_save_path_ep / key / f"goal_scores_{self.step}.txt", 'w') as f:
                        for score in goal_scores:
                            f.write(f"{score}\n")
                elif key=='vlm_response':
                    vlm_response = trace_asset.get('vlm_response', None)
                    if vlm_response is None: continue
                    # save in json format
                    with open(self.assets_save_path_ep / key / f"vlm_response_{self.step}.json", 'w') as f:
                        json.dump(vlm_response, f, indent=4)
                elif key=='goal_poses':
                    goal_poses = trace_asset.get('goal_poses', None)
                    if goal_poses is None: continue
                    with open(self.assets_save_path_ep / key / f"goal_poses_{self.step}.json", 'w') as f:
                        json.dump([pose.tolist() for pose in goal_poses], f, indent=4)
