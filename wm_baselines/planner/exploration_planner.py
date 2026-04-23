from typing import Any, Dict, Union
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from omegaconf import DictConfig
from copy import deepcopy
from wm_baselines.agent.perception.occupancy import OccupancyMap
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.utils.planning_utils import rotation_angle
from wm_baselines.utils.common_utils import with_timing
from wm_baselines.planner.planning.path_planning import PATH_PLANNING_REGISTRY, path_to_trajectory
from wm_baselines.planner.planning.goal_sampling import GOAL_SAMPLING_REGISTRY

class ExplorationPlanner(BasePlanner):
    """Exploration planner using frontier-based exploration strategy.

    Responsibilities:
    - hold config reference
    - implement plan() to select actions that explore unknown areas
    """

    def __init__(
        self, 
        goal_sampling_strategy: str,
        path_planning_algorithm: str,
        action_name: str,
        adjacent_angle: float,
        adjacent_distance: float,
        stuck_rotate_times: int = 11,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.explored_map = None
        self.current_asset = None
        self.current_goal = None
        self.current_path = []
        self.current_trajectory = []
        self.path_keypoints = []
        self.replan = True  # whether to replan using the most recent assets
        self.random_free = False  # whether to sample random free space as goal
        self.pose_check_dict = {}
        self.goal_sampling_strategy = goal_sampling_strategy
        assert self.goal_sampling_strategy in GOAL_SAMPLING_REGISTRY, f"Unknown goal sampling strategy {self.goal_sampling_strategy}"
        self.path_planning_algorithm = path_planning_algorithm
        assert self.path_planning_algorithm in PATH_PLANNING_REGISTRY, f"Unknown path planning algorithm {self.path_planning_algorithm}"
        self.action_name = action_name
        self.adjacent_angle = adjacent_angle
        self.adjacent_distance = adjacent_distance
        self.step = -1
        self.current_goal_steps = 0
        self.current_path_len = 0
        self._metrics = {
            "planning_time": 0.0,
        }
        self.stuck_rotate_times = stuck_rotate_times
        self.rotate_count = 0

    def reset(self):
        """Reset the planner state."""
        self.explored_map = None
        self.current_goal = None
        self.current_path = []
        self.current_trajectory = []
        self.path_keypoints = []
        self.replan = True
        self.random_free = False
        self.pose_check_dict = {}
        self.current_asset = None
        self.step = -1
        self.current_goal_steps = 0
        self.current_path_len = 0
        self._metrics = {
            "planning_time": 0.0,
        }
        self.rotate_count = 0

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

        if self.replan or self.random_free or len(self.current_trajectory)==0:
            print("Replanning...")
            self.current_path, exe_time = self.plan()
            self._metrics["planning_time"] += exe_time
            if self.current_path is None or len(self.current_path) == 0:
                self.current_trajectory = []
                self.path_keypoints = []
                print("No more valid path found")
                trace_asset = {
                    # "pcd": current_asset["pcd"],
                    "rgb": current_asset["rgb"],
                    "occupancy": deepcopy(current_asset["occupancy"]),
                    "path": None,
                    "goal": self.current_goal if self.current_goal is not None else None,
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
            # "pcd": current_asset["pcd"],
            "rgb": current_asset["rgb"],
            "occupancy": deepcopy(current_asset["occupancy"]),
            "path": path_keypoints if len(path_keypoints)>0 else None,
            "goal": self.current_goal if self.current_goal is not None else None,
        }

        return action, trace_asset

    @with_timing
    def plan(self) -> Dict[str, Any]:
        """Given the current observation and assets, return the next action dict."""
        occ: OccupancyMap = self.current_asset["occupancy"]
        current_pose: Union[Tensor, np.ndarray] = self.current_asset["pose"]  # (4, 4)
        start = current_pose[:3, 3].cpu().numpy() if isinstance(current_pose, Tensor) else current_pose[:3, 3]  # (3,)
        # sample goal
        goals, forwards = GOAL_SAMPLING_REGISTRY[self.goal_sampling_strategy](occ, current_pose)
        if self.random_free or (goals is None or len(goals) == 0):
            goals, forwards = GOAL_SAMPLING_REGISTRY["random_free"](occ, current_pose)
            self.random_free = False  # reset random free goal sampling
        if goals is None or len(goals) == 0:
            return []
        print("Goals", goals)
        self.current_goal = goals[0]  # (x, y, z)
        self.current_forward = forwards[0]  # (3,)
        # plan path
        current_path = PATH_PLANNING_REGISTRY[self.path_planning_algorithm](occ, start, self.current_goal)
        if current_path is None or len(current_path) == 0:
            print("Path planning failed, using random walk fallback")
            current_path = []
            self.random_free = True  # force random free goal sampling next time
            z_idx = self.current_asset["pose"][:, 2][:3]  # current forward vector
            t_idx = self.current_asset["pose"][:, 3][:3]  # current translation

        self.current_path = current_path
        return self.current_path

    def _update_replan(self, force: bool = False) -> None:
        """Update whether to replan at each step."""
        if force:
            self.replan = True
            return

        occ = self.current_asset["occupancy"]
        current_pose = self.current_asset["pose"]  # (4, 4)
        start = current_pose[:3, 3].cpu().numpy() if isinstance(current_pose, Tensor) else current_pose[:3, 3]  # (3,)
        if self.current_goal is not None:
            current_path = PATH_PLANNING_REGISTRY[self.path_planning_algorithm](occ, start, self.current_goal)
            if current_path is None or len(current_path) == 0:
                self.replan = True # if path is blocked, replan
        if self.current_goal_steps >= 2*self.current_path_len:  # max steps towards one goal
            self.replan = True  # replan after too many steps towards one goal

    def _detect_agent_stuck(self) -> bool:
        """Check if the agent is stuck."""
        # Check if the agent has not moved significantly from the last pose
        if self.step == 0:
            self.pose_check_dict["last_pose"] = self.current_asset["pose"]
            return
        if self.step % 5 != 0:
            return
        position_diff = np.linalg.norm(self.current_asset["pose"][:3, 3] - self.pose_check_dict["last_pose"][:3, 3])
        rotation_diff = rotation_angle(self.current_asset["pose"][:3, 2], self.pose_check_dict["last_pose"][:3, 2])
        stuck = position_diff < 1e-6 and rotation_diff < 1e-6
        if stuck:
            print(f"Agent seems to be stuck at step {self.step}, forcing replan")
            self.random_free = True  # force random free goal sampling next time
            self.rotate_count = 0  # reset rotate count

        self.pose_check_dict["last_pose"] = self.current_asset["pose"]
