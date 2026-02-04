from typing import Any, Dict, Union, List
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from omegaconf import DictConfig
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
from belief_baselines.agent.perception.occupancy import OccupancyMap
from belief_baselines.planner.base_planner import BasePlanner
from belief_baselines.utils.planning_utils import rotation_angle, goals_and_forwards_to_poses
from belief_baselines.utils.common_utils import with_timing
from belief_baselines.world_model.base_world_model import BaseWorldModel
from belief_baselines.agent.vlm.vlm import VLM


ACTION_SPACE = [
    "move_forward",
    "turn_left",
    "turn_right",
]

class CEMPlanner(BasePlanner):
    """Cross-Entropy Method (CEM) planner for action selection."""

    def __init__(
        self,
        vlm: VLM,
        horizon: int,
        num_sequences: int = 10,
        move_step: float = 0.3,
        rotation_step: float = 10.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.current_asset = None
        self.step = -1
        self.horizon = horizon
        self.num_sequences = num_sequences
        self.move_step = move_step
        self.rotation_step = rotation_step
        self.vlm = vlm
        self.world_model : BaseWorldModel = None  # to be set externally
        self.current_actions: List[str] = []
        self.current_trajectory: List[str] = []
        self._metrics = {
            "planning_time": 0.0,
            "vlm_input_tokens": 0,
            "vlm_output_tokens": 0,
        }

    def reset(self):
        """Reset the planner state."""
        self.current_asset = None
        self.step = -1
        self.current_actions = []
        self.current_trajectory = []
        self._metrics = {
            "planning_time": 0.0,
            "vlm_input_tokens": 0,
            "vlm_output_tokens": 0,
        }

    def get_next_action(self, current_asset: Dict[str, Any], current_step: int, force_update: bool = False) -> Dict[str, Any]:
        """Get the next action based on current assets and step count."""
        if current_asset is not None and current_step > self.step:
            self.set_state(current_asset)
            self.step += 1
            assert self.step == current_step, f"step mismatch: {self.step} vs {current_step}"
        if 'pose' in current_asset:
            self.current_asset["pose"] = current_asset["pose"]

        if len(self.current_actions) == 0:
            self.current_actions, exe_time = self.plan()
            self.current_trajectory = deepcopy(self.current_actions)
            self._metrics["planning_time"] += exe_time
        
        next_action = self.current_actions.pop(0)

        action = {"action_name": next_action, "args": {}}

        trace_asset = {
            # "pcd": current_asset["pcd"],
            "rgb": current_asset["rgb"],
            "occupancy": deepcopy(current_asset["occupancy"]),
            "goal": self.current_goal if self.current_goal is not None else None,
            "all_goals": self.all_goals if len(self.all_goals) > 0 else None,
            "goal_images": self.current_asset.get("goal_images", None),
            "vlm_scores": self.current_asset.get("vlm_scores", None),
            "vlm_response": self.current_asset.get("vlm_response", None),
            "goal_poses": self.current_asset.get("goal_poses", None),
        }
        return action, trace_asset
    
    @with_timing
    def plan(self) -> Dict[str, Any]:
        """Plan the next action using CEM."""
        current_pose: np.ndarray = self.current_asset["pose"]  # (4, 4)
        initial_pose: np.ndarray = self.current_asset["initial_pose"]  # (4, 4)
        object_name: str = self.current_asset["object_name"]
        # Sample random action sequences
        action_sequences = np.random.choice(
            ACTION_SPACE,
            size=(self.num_sequences, self.horizon)
        )
        # for each action sequence, calculate the position and forward at the end of the horizon
        goals, forwards = self._get_goal_poses(action_sequences, current_pose)
        # use VLM to select the best goal
        best_idx = self.select_goal_vlm(goals, forwards, initial_pose, object_name)
        best_action_sequence = action_sequences[best_idx]
        self.all_goals = goals
        self.current_goal = goals[best_idx]
        actions = best_action_sequence.tolist()
        return actions

    def select_goal_vlm(self, goals: List[np.ndarray], forwards: List[np.ndarray], initial_pose: Union[Tensor, np.ndarray], object_name: str) -> int:
        """Use VLM to select goal based on images and object name."""
        assert self.world_model is not None, "World model must be set for VLMPlanner"
        images = self.world_model.render_goal_images(goals, forwards, initial_pose).rgb  # list of np arrays
        goal_poses = goals_and_forwards_to_poses(goals, forwards)
        self.current_asset["goal_images"] = images
        self.current_asset["goal_poses"] = goal_poses
        response = self.vlm.prompt_score_obj_images(images, object_name)
        results = response['parsed']
        self.current_asset["vlm_scores"] = [res[1] for res in results]
        scores = [res[1] for res in results]  # get scores
        best_idx = int(np.argmax(scores)) if len(scores) > 0 else 0
        best_idx = min(best_idx, len(goals)-1)  # ensure within bounds
        self.current_asset["vlm_response"] = response
        self._metrics["vlm_input_tokens"] += response.get("num_input_tokens", 0)
        self._metrics["vlm_output_tokens"] += response.get("num_output_tokens", 0)
        return best_idx
    
    def set_world_model(self, world_model: BaseWorldModel) -> None:
        """Set the world model for the planner."""
        self.world_model = world_model
    
    def _get_goal_poses(self, action_sequences: np.ndarray, current_pose: np.ndarray) -> (List[np.ndarray], List[np.ndarray]):
        """Given action sequences and current pose, compute goal positions and forwards."""
        goals = []
        forwards = []
        for seq in action_sequences:
            pose = current_pose.copy()
            for action in seq:
                if action == "move_forward":
                    t_forward = np.array([0.0, 0.0, self.move_step], dtype=np.float32)
                    T_forward = np.eye(4, dtype=np.float32)
                    T_forward[:3, 3] = t_forward.astype(np.float32)
                    pose = pose @ T_forward
                elif action == "turn_left":
                    angle = -self.rotation_step
                    R_aw_left = R.from_euler('xyz', [0.0, angle, 0.0], degrees=True).as_matrix()
                    T_aw_left = np.eye(4, dtype=np.float32)
                    T_aw_left[:3, :3] = R_aw_left.astype(np.float32)
                    pose = pose @ T_aw_left
                elif action == "turn_right":
                    angle = self.rotation_step
                    R_aw_right = R.from_euler('xyz', [0.0, angle, 0.0], degrees=True).as_matrix()
                    T_aw_right = np.eye(4, dtype=np.float32)
                    T_aw_right[:3, :3] = R_aw_right.astype(np.float32)
                    pose = pose @ T_aw_right
            goal = pose[:3, 3]
            forward = pose[:3, 2]
            goals.append(goal)
            forwards.append(forward)
        return goals, forwards