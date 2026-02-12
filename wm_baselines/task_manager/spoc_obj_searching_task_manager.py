import os
import gzip
import json
import jsonlines
import time
from omegaconf import DictConfig, OmegaConf
from environment.stretch_controller import StretchController
from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from spoc_utils.data_generation_utils.navigation_utils import is_any_object_sufficiently_visible_and_in_center_frame
from spoc_utils.embodied_utils import find_object_node
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from copy import deepcopy
from wm_baselines.task_manager.base_task_manager import BaseTaskManager


class SpocObjSearchingTaskManager(BaseTaskManager):
    def __init__(self, embodied_config: DictConfig, stretch_controller: StretchController, **kwargs):
        super().__init__(embodied_config)
        self.episode_root = embodied_config.episode_root
        self.subset_type = embodied_config.get("subset_type", None)  # "room" or "length"
        self.subset = embodied_config.get("subset", None)
        self.close_enough_distance = embodied_config.close_enough_distance
        self.alignment_threshold = embodied_config.alignment_threshold
        self.max_steps = embodied_config.max_steps
        self.stretch_controller = stretch_controller

        self.episode_list = sorted([p for p in Path(self.episode_root).glob("*/") if p.is_dir()])
        self.episodes = [self._load_episode(p) for p in self.episode_list]
        self._handle_subset()
        self.current_episode_index = -1
        self.current_target_obj = None
        self.current_target_info = None
        self.current_house_index = None
        self.initial_distance_to_goal = None

    @property
    def distance_to_goal(self):
        assert self.current_target_info is not None, "Run reset() before accessing distance_to_goal"
        goal_pos = self.current_target_info['position']
        agent_pos = self.stretch_controller.controller.last_event.metadata["cameraPosition"]
        dx = goal_pos['x'] - agent_pos['x']
        dy = goal_pos['y'] - agent_pos['y']
        dz = goal_pos['z'] - agent_pos['z']
        distance_to_goal = (dx**2 + dy**2 + dz**2)**0.5
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
        return f"{self.current_episode_index}_{self.episodes[self.current_episode_index]['house_index']}_{self.episodes[self.current_episode_index]['target_object']}"

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
        self.current_house_index = episode["house_index"]
        initial_position = episode["initial_position"]
        initial_rotation = episode["initial_rotation"]
        # initial_position = episode["final_position"]
        # initial_rotation = episode["final_rotation"]
        print(f"Starting episode {self.current_episode_index}: House {self.current_house_index}, Target Object: {self.current_target_obj}")
        house_data = self._load_house_from_prior(self.current_house_index)
        self.stretch_controller.reset(house_data)
        self.stretch_controller.step(
            action="TeleportFull",
            position=initial_position,
            rotation=initial_rotation,
            horizon=0.0,  # Level camera view
            standing=True  # Agent is standing
        )
        scene_graph = self.stretch_controller.current_scene_json['objects']

        self.current_target_info = find_object_node(scene_graph, self.current_target_obj)
        assert self.current_target_info is not None, f"Target object {self.current_target_obj} not found in scene graph"
        self.initial_distance_to_goal = self.distance_to_goal

        super().reset(idx)

    def is_done(self):
        visible = is_any_object_sufficiently_visible_and_in_center_frame(
            self.stretch_controller,
            [self.current_target_obj],
            alignment_threshold=self.alignment_threshold
        )
        close_enough = self.distance_to_goal < self.close_enough_distance
        done = visible and close_enough
        return done

    def get_final_log(self, metrics: dict = None):
        episode = self.episodes[self.current_episode_index]
        final_log = {
            "episode_index": self.current_episode_index,
            "house_index": self.current_house_index,
            "room_annotation": episode["room_annotation"],
            "length_annotation": episode["length_annotation"],
            "target_object": self.current_target_obj,
            "num_steps": self.current_step,
            "success": self.is_done(),
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
        object_name = object_name.split('|')[0]
        if "Obja" in object_name:
            object_name = object_name.split('Obja')[1]
        object_name = object_name.lower()
        return object_name

    def _load_episode(self, episode_path: Path):
        """Load episode info (episode_index, house_id, target_object) 
        from the given path."""
        metadata_path = episode_path / "trajectory_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if "," in metadata["object_id"]:
            metadata["object_id"] = metadata["object_id"].split(",")[0]  # take the first object if multiple
        episode = {
            "house_index": int(metadata["house_id"]),
            "target_object": metadata["object_id"],
            "oracle_length": metadata["frames"],
            "room_annotation": metadata.get("room_annotation", ""),
            "length_annotation": metadata.get("length_annotation", ""),
            "distance_traveled": metadata.get("distance_traveled", -1.0),
            "angle_turned": metadata.get("angle_turned", -1.0),
            "initial_position": metadata["initial_position"],
            "initial_rotation": metadata["initial_rotation"],
            "final_position": metadata["final_position"],
            "final_rotation": metadata["final_rotation"],
        }
        return episode

    def _handle_subset(self):
        if self.subset_type is None or self.subset is None:
            return
        if self.subset_type == "room":
            filtered_episodes = [ep for ep in self.episodes if self.subset in ep["room_annotation"]]
        elif self.subset_type == "length":
            filtered_episodes = [ep for ep in self.episodes if self.subset in ep["length_annotation"]]
        else:
            raise ValueError(f"Unknown subset_type {self.subset_type}, must be 'room' or 'length'")
        
        if len(filtered_episodes) == 0:
            raise ValueError(f"No episodes found for subset {self.subset} of type {self.subset_type}")
        
        print(f"Filtered episodes from {len(self.episodes)} to {len(filtered_episodes)} using subset_type={self.subset_type}, subset={self.subset}")
        self.episodes = filtered_episodes

    def _load_house_from_prior(self, house_index: int):
        """
        Load a house directly from local JSONL files instead of using the prior dataset
        
        Parameters:
        house_index (int): Index of the house to load (default: 0)
        
        Returns:
        dict: House data as a dictionary
        """
        
        print(f"Loading house at index {house_index} from local dataset...")
        
        # If direct file not found, try to load from JSONL files
        for split in ["val"]:
            houses_path = os.path.join(OBJAVERSE_HOUSES_DIR, f"{split}.jsonl.gz")
            
            if not os.path.exists(houses_path):
                print(f"Warning: {houses_path} does not exist, trying next split")
                continue
            
            print(f"Load from {houses_path}")
            try:
                # Manual approach using gzip and line-by-line JSON parsing to handle errors
                current_index = 0
                with gzip.open(houses_path, 'rt', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            # Skip empty lines
                            line = line.strip()
                            if not line:
                                continue
                            
                            # Parse JSON
                            house = json.loads(line)
                            
                            # Check if this is the house we want
                            if current_index == house_index:
                                print(f"Successfully loaded house at index {house_index} from {split} split (line {line_num})")
                                if 'id' in house:
                                    print(f"House ID: {house['id']}")
                                return house
                            
                            current_index += 1
                            
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON at line {line_num}, skipping")
                        except Exception as e:
                            print(f"Warning: Error processing line {line_num}: {e}, skipping")
                
                print(f"Reached end of file {houses_path} after processing {current_index} houses, but didn't find index {house_index}")
                
            except Exception as e:
                print(f"Error reading {houses_path}: {e}")
        
        # Fallback to a simple house creation approach
        print(f"Could not find house with index {house_index}, creating a default house")
        
        # Create a minimal default house that will work with the StretchController
        default_house = {
            "id": f"default_{house_index}",
            "objects": [],
            "rooms": [],
            "scene_bounds": {
                "center": {"x": 0, "y": 0, "z": 0},
                "size": {"x": 10, "y": 3, "z": 10}
            }
        }
        
        return default_house
    