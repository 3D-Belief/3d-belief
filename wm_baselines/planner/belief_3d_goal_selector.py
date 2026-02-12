from typing import Any, Dict, Union, List
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from omegaconf import DictConfig
from copy import deepcopy
from rollout_utils import visualize_semantic_query_intensity_map
from wm_baselines.agent.perception.occupancy import OccupancyMap
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.utils.planning_utils import rotation_angle, goals_and_forwards_to_poses
from wm_baselines.utils.common_utils import with_timing
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.planning.path_planning import PATH_PLANNING_REGISTRY, path_to_trajectory
from wm_baselines.planner.planning.goal_sampling import GOAL_SAMPLING_REGISTRY
from wm_baselines.planner.exploration_planner import ExplorationPlanner

class Belief3DModelGoalSelector(ExplorationPlanner):
    """A planner that uses 3d belief model built-in semantics for goal selection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.world_model: BaseWorldModel = None  # to be set externally
    
    def reset(self):
        super().reset()

    def get_next_action(self, current_asset: Dict[str, Any], current_step: int, force_update: bool = False) -> Dict[str, Any]:
        """Get the next action."""
        if current_asset is not None and current_step > self.step:
            self.set_state(current_asset)
            self.step += 1
            assert self.step == current_step, f"step mismatch: {self.step} vs {current_step}"
        self.current_goal_steps += 1
        self._update_replan(force=force_update)
        self._detect_agent_stuck()
        if 'pose' in current_asset:
            self.current_asset["pose"] = current_asset["pose"]

        if self.random_free:
            self.current_trajectory = []
            self.path_keypoints = []
            print("Agent is stuck, performing in-place rotation to free itself")
            trace_asset = {
                # "pcd": current_asset["pcd"],
                "rgb": current_asset["rgb"],
                "occupancy": deepcopy(current_asset["occupancy"]),
                "path": None,
                "goal": self.current_goal if self.current_goal is not None else None,
                "all_goals": self.all_goals if len(self.all_goals) > 0 else None,
                "goal_images": self.current_asset.get("goal_images", None),
                "goal_semantics": self.current_asset.get("goal_semantics", None),
                "goal_scores": self.current_asset.get("goal_scores", None),
                "goal_poses": self.current_asset.get("goal_poses", None),
            }
            self.rotate_count += 1

            if self.rotate_count <= 2: # first two times move back
                ret= ({"action_name": "move_back", "args": {}}, trace_asset)
            else:
                ret = ({"action_name": "turn_right", "args": {}}, trace_asset)
            
            if self.rotate_count >= self.stuck_rotate_times:
                self.random_free = False  # reset after rotating enough times
                self.rotate_count = 0
            
            return ret

        if self.replan or self.random_free or len(self.current_trajectory)==0:
            print("Replanning...")
            self.current_path, exe_time = self.plan()
            self._metrics["planning_time"] += exe_time
            if self.current_path is None or len(self.current_path) == 0:
                self.current_trajectory = []
                self.path_keypoints = []
                print("No more valid path found")
                trace_asset = {
                    "rgb": current_asset["rgb"],
                    "occupancy": deepcopy(current_asset["occupancy"]),
                    "path": None,
                    "goal": self.current_goal if self.current_goal is not None else None,
                    "all_goals": self.all_goals if len(self.all_goals) > 0 else None,
                    "goal_images": self.current_asset.get("goal_images", None),
                    "goal_semantics": self.current_asset.get("goal_semantics", None),
                    "goal_scores": self.current_asset.get("goal_scores", None),
                    "goal_poses": self.current_asset.get("goal_poses", None),
                }
                return {"action_name": "turn_right", "args": {}}, trace_asset
            self.current_trajectory, self.path_keypoints = path_to_trajectory(self.current_path, occ=self.current_asset["occupancy"], ref_pose=self.current_asset["initial_pose"])
            self.replan = False
            self.current_goal_steps = 0
            self.current_path_len = len(self.current_path)

        next_pose = self.current_trajectory.pop(0)  # (x, y, z)
        path_keypoints = self.path_keypoints.copy()      # or lst[:] for a shallow copy
        self.path_keypoints.pop(0)
        action = {"action_name": self.action_name, "args": {"target_pose": next_pose}}

        trace_asset = {
            "rgb": current_asset["rgb"],
            "occupancy": deepcopy(current_asset["occupancy"]),
            "path": path_keypoints if len(path_keypoints)>0 else None,
            "goal": self.current_goal if self.current_goal is not None else None,
            "all_goals": self.all_goals if len(self.all_goals) > 0 else None,
            "goal_images": self.current_asset.get("goal_images", None),
            "goal_semantics": self.current_asset.get("goal_semantics", None),
            "goal_scores": self.current_asset.get("goal_scores", None),
            "goal_poses": self.current_asset.get("goal_poses", None),
        }

        return action, trace_asset

    @with_timing
    def plan(self) -> Dict[str, Any]:
        """Given the current observation and assets, return the next action dict."""
        occ: OccupancyMap = self.current_asset["occupancy"]
        current_pose: Union[Tensor, np.ndarray] = self.current_asset["pose"]  # (4, 4)
        initial_pose: Union[Tensor, np.ndarray] = self.current_asset["initial_pose"]  # (4, 4)
        object_name: str = self.current_asset["object_name"]
        start = current_pose[:3, 3].cpu().numpy() if isinstance(current_pose, Tensor) else current_pose[:3, 3]  # (3,)
        # sample goal
        goals, forwards = GOAL_SAMPLING_REGISTRY[self.goal_sampling_strategy](occ, current_pose)
        if goals is None or len(goals) == 0:
            self.all_goals = []
            self.current_goal = None
            return []  # no valid goal found
        else:
            self.all_goals = goals
            best_idx = self.select_goal_semantic(goals, forwards, initial_pose, object_name)
            self.current_goal = goals[best_idx]  # (x, y, z)
            self.current_forward = forwards[best_idx]  # (3,)
        # plan path
        current_path = PATH_PLANNING_REGISTRY[self.path_planning_algorithm](occ, start, self.current_goal)
        if current_path is None or len(current_path) == 0:
            print("Path planning failed, using random walk fallback")
            current_path = PATH_PLANNING_REGISTRY["random_walk"](occ, start, self.current_goal)
            self.replan = True  # force replan next time
        self.current_path = current_path
        return self.current_path

    def select_goal_semantic(self, goals: List[np.ndarray], forwards: List[np.ndarray], initial_pose: Union[Tensor, np.ndarray], object_name: str) -> int:
        """Use 3d belief built-in semantic to select goal based on images and object name."""
        assert self.world_model is not None, "World model must be set for Belief3DModelGoalSelector"
        object_name: str = self.current_asset["object_name"]
        render_output = self.world_model.render_goal_images(goals, forwards, initial_pose, query_label=object_name)  # list of np arrays
        images = render_output.rgb  # list of np arrays
        semantics = render_output.semantic  # list of np arrays
        semantics_viz = [np.ascontiguousarray(visualize_semantic_query_intensity_map(semantic[0])) for semantic in semantics]  # list of np arrays
        goal_poses = goals_and_forwards_to_poses(goals, forwards)
        self.current_asset["goal_images"] = images
        self.current_asset["goal_semantics"] = semantics_viz
        self.current_asset["goal_poses"] = goal_poses
        self.current_asset["goal_scores"] = [np.max(semantics[i]) for i in range(len(semantics))]
        best_idx = int(np.argmax(self.current_asset["goal_scores"])) if len(self.current_asset["goal_scores"]) > 0 else 0
        best_idx = min(best_idx, len(goals)-1)  # ensure within bounds
        return best_idx
    
    def set_world_model(self, world_model: BaseWorldModel) -> None:
        """Set the world model for the planner."""
        self.world_model = world_model