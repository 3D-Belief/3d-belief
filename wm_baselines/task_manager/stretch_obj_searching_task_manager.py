import os
import gzip
import json
import jsonlines
import time
from omegaconf import DictConfig
from stretch.agent.zmq_client import HomeRobotZmqClient
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from copy import deepcopy
from wm_baselines.task_manager.base_task_manager import BaseTaskManager
from wm_baselines.agent.vlm.object_detection import segment_label_with_gemini


class StretchObjSearchingTaskManager(BaseTaskManager):
    def __init__(self, embodied_config: DictConfig, stretch_controller: HomeRobotZmqClient, **kwargs):
        super().__init__(embodied_config)
        self.episode_root = embodied_config.episode_root
        self.close_enough_distance = embodied_config.close_enough_distance
        self.alignment_threshold = embodied_config.alignment_threshold
        self.max_steps = embodied_config.max_steps
        self.stretch_controller = stretch_controller

        self.episode_list = sorted([p for p in Path(self.episode_root).glob("*/") if p.is_dir()])
        self.episodes = [self._load_episode(p) for p in self.episode_list]
        # self._handle_subset()
        self.current_episode_index = -1
        self.current_target_obj = None
        self.current_target_position = None
        self.initial_distance_to_goal = None

    @property
    def distance_to_goal(self):
        assert self.current_target_position is not None, "Run reset() before accessing distance_to_goal"
        goal_pos = self.current_target_position
        x, y, _ = self.stretch_controller.get_base_pose()
        dx = goal_pos['x'] - x
        dy = goal_pos['y'] - y
        distance_to_goal = (dx**2 + dy**2)**0.5
        return distance_to_goal
    
    @property
    def distance_traveled(self):
        return self._distance_traveled
    
    @property
    def angle_turned(self):
        return self._angle_turned

    @property
    def current_step(self):
        return self._current_step
    
    @property
    def current_ep_name(self):
        return f"{self.current_episode_index}_{self.episodes[self.current_episode_index]['target_object']}"

    def reset(self, idx=None):
        if idx is not None:
            if idx < 0 or idx >= len(self.episodes):
                raise IndexError(f"Index {idx} is out of bounds for episodes list.")
            self.current_episode_index = idx
        else:
            self.current_episode_index += 1

        if self.current_episode_index >= len(self.episodes):
            print("All episodes completed.")
            return
        episode = self.episodes[self.current_episode_index]
        self.current_target_obj = episode["target_object"]
        print(f"Starting episode {self.current_episode_index}: Target Object: {self.current_target_obj}")
        self.current_target_position = episode["object_position"]
        assert self.current_target_position is not None, f"Target object {self.current_target_obj} not found"
        self.initial_distance_to_goal = self.distance_to_goal

        super().reset(idx)

    def is_done(self, rgb):
        detections, annotated = segment_label_with_gemini(image_np=rgb, label=self.current_target_obj, return_annotated=True)
        if len(detections) == 0 or detections.mask is None:
            return False
        else:
            return True
        box = detections.xyxy[0]
        x_min, y_min, x_max, y_max = box.tolist()
        center = (int((x_min + x_max) / 2), int((y_min + y_max) / 2))
        # check whether center is within the (h/2, w/2) threshold
        h, w, _ = rgb.shape
        center_threshold = 0.4 * (w + h) / 2
        dist_to_center = ((center[0] - w / 2) ** 2 + (center[1] - h / 2) ** 2) ** 0.5
        ret = dist_to_center < center_threshold
        return ret

    def get_final_log(self, metrics: dict = None):
        episode = self.episodes[self.current_episode_index]
        final_log = {
            "episode_index": self.current_episode_index,
            "target_object": self.current_target_obj,
            "num_steps": self.current_step,
            "success": self.is_done(metrics['rgb']),
            "initial_distance_to_goal": self.initial_distance_to_goal,
            "final_distance_to_goal": self.distance_to_goal,
            "distance_traveled": self.distance_traveled,
            "angle_turned": self.angle_turned,
            "oracle_length": episode["oracle_length"],
            "oracle_distance_traveled": episode["distance_traveled"],
            "oracle_angle_turned": episode["angle_turned"],
            "time_taken": time.time() - self.start_time,
        }
        if metrics is not None:
            final_log.update(metrics)
        return final_log
    
    def fix_object_name(self, object_name: str) -> str:
        """Fix object name to match the naming convention in the scene graph."""
        return object_name

    def _load_episode(self, episode_path: Path):
        """Load episode info (episode_index, target_object) 
        from the given path."""
        metadata_path = episode_path / "metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        episode = {
            "target_object": metadata["object_id"],
            "object_position": metadata["object_position"],
            "oracle_length": metadata["frames"],
            "distance_traveled": metadata.get("distance_traveled", -1.0),
            "angle_turned": metadata.get("angle_turned", -1.0),
        }
        return episode
