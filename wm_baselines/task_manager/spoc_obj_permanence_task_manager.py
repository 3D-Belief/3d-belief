import os
import gzip
import json
import jsonlines
import time
import numpy as np
from typing import Any, Dict, Tuple
import open3d as o3d
import cv2
import torch
from omegaconf import DictConfig, OmegaConf
from environment.stretch_controller import StretchController
from spoc_utils.constants.stretch_initialization_utils import STRETCH_ENV_ARGS
from spoc_utils.constants.objaverse_data_dirs import OBJAVERSE_HOUSES_DIR
from spoc_utils.data_generation_utils.navigation_utils import is_any_object_sufficiently_visible_and_in_center_frame
from spoc_utils.embodied_utils import find_object_node
from scipy.spatial.transform import Rotation as R
from skimage.filters import gaussian
from skimage.segmentation import flood
from skimage.morphology import disk, binary_opening, binary_closing, remove_small_objects, binary_dilation
from skimage.measure import label, regionprops, perimeter
from scipy.ndimage import binary_fill_holes

from PIL import Image
from pathlib import Path
from copy import deepcopy
from wm_baselines.task_manager.base_task_manager import BaseTaskManager
from wm_baselines.agent.perception.metrics import Box3D, box3d_from_aabb
from wm_baselines.agent.perception.camera import Camera
from wm_baselines.utils.planning_utils import rotation_angle
from wm_baselines.agent.vlm.vlm import VLM
from wm_baselines.agent.perception.occupancy import OccupancyMap
from wm_baselines.world_model.oracle_depth_model import OracleDepthModel

class SpocObjPermanenceTaskManager(BaseTaskManager):
    def __init__(self, embodied_config: DictConfig, stretch_controller: StretchController, **kwargs):
        super().__init__(embodied_config)
        self.episode_root = embodied_config.episode_root
        self.adjacent_angle = embodied_config.adjacent_angle
        self.adjacent_distance = embodied_config.adjacent_distance
        self.occupancy_resolution = embodied_config.occupancy_resolution
        self.occupancy_obstacle_height_thresh = embodied_config.occupancy_obstacle_height_thresh
        self.num_steps = None  # to be set during reset
        self.stretch_controller = stretch_controller

        self.episode_list = sorted([p for p in Path(self.episode_root).glob("*/") if p.is_dir()])
        self.episodes = [self._load_episode(p) for p in self.episode_list]
        self.observations = None
        self.imagination_poses = None
        self.imagination_key_poses = None
        self.imagination_key_rgbs = None
        self.imagination_key_depths = None
        self.done = False
        self.current_episode_index = -1
        self.current_house_index = None
    
    @property
    def current_step(self):
        return self._current_step
    
    @property
    def current_ep_name(self):
        # get a random string for the current episode name
        return f"{self.current_episode_index}_{self.episodes[self.current_episode_index]['house_index']}_{time.time():.0f}"

    def set_camera(self, camera: Camera):
        self.camera = camera
    
    def set_vlm(self, vlm: VLM):
        self.vlm = vlm

    def get_observation(self):
        """Get the current observation of the episode."""
        obs = self.observations[self._current_step]
        imagination_poses = [self.observations[i]['pose'] for i in range(self._current_step, len(self.observations))]
        self.imagination_poses = imagination_poses
        self.imagination_key_poses, imagination_key_indices = self._extract_key_poses(imagination_poses)
        self.imagination_key_rgbs = [self.observations[self._current_step + idx]['rgb'] for idx in imagination_key_indices]
        self.imagination_key_depths = [self.observations[self._current_step + idx]['depth'] for idx in imagination_key_indices]
        self._current_step += 1
        # if self._current_step == len(self.observations):
        #     self.done = True
        self.done = True
        return obs

    def get_imagination_poses(self):
        """Get the imagination poses for the current step."""
        return self.imagination_poses
    
    def _extract_key_poses(self, poses):
        key_poses = []
        key_indices = []
        z_start = poses[0][:, 2][:3]
        t_start = poses[0][:, 3][:3]
        z_previous = z_start
        t_previous = t_start
        num_frames = len(poses)
        for idx in range(1, num_frames):
            current_pose = poses[idx]
            z_idx = current_pose[:, 2][:3]  # current forward vector
            t_idx = current_pose[:, 3][:3]  # current translation
            angle = rotation_angle(z_previous, z_idx)
            distance = np.linalg.norm(t_idx - t_previous)
            if angle > self.adjacent_angle or distance > self.adjacent_distance or idx==num_frames-1: # must include the last
                key_poses.append(current_pose)
                key_indices.append(idx)
                z_previous = z_idx
                t_previous = t_idx
        return key_poses, key_indices

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
        self.current_house_index = episode["house_index"]
        self.num_steps = episode["oracle_length"]
        self.done = False
        self.imagination_poses = []
        self.imagination_key_poses = []
        self.imagination_key_rgbs = []
        self.imagination_key_depths = []
        self._current_step = 0
        self._load_observation()

        print(f"Starting episode {self.current_episode_index}: House {self.current_house_index}")
        house_data = self._load_house_from_prior(self.current_house_index)
        super().reset(idx)

    def is_done(self):
        return self.done

    def get_final_log(self, metrics: dict = None):
        episode = self.episodes[self.current_episode_index]
        final_log = {
            "episode_index": self.current_episode_index,
            "house_index": self.current_house_index,
            "num_steps": self.current_step,
            "success": self.is_done(),
            "oracle_length": episode["oracle_length"],
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
    
    def lower_object_name(self, object_name: str) -> str:
        """Convert object name to lowercase and handle special cases."""
        object_name = object_name.lower()
        return object_name

    def _load_episode(self, episode_path: Path):
        """Load episode info from the given path."""
        metadata_path = episode_path / "trajectory_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        episode = {
            "episode_path": str(episode_path),
            "house_index": int(metadata["house_id"].split("_")[-1]),
            "oracle_length": metadata["frames"],
        }
        return episode
    
    def _load_observation(self):
        """Load observation of the current episode."""
        episode = self.episodes[self.current_episode_index]
        # Load RGB images
        rgb_path = Path(episode["episode_path"]) / f"rgb"
        rgb_files = sorted(list(rgb_path.glob("*.png")))
        # load all rgb images using PIL
        rgbs = [np.array(Image.open(str(f)).convert("RGB")) for f in rgb_files]
        # Load Depth images
        depth_path = Path(episode["episode_path"]) / f"depth"
        depth_files = sorted(list(depth_path.glob("*.png")))
        # load all depth images with scaling to meters
        depths = []
        for f in depth_files:
            depth_raw = np.array(Image.open(str(f)))
            depth_in_meters = depth_raw.astype(np.float32) / 1000.0  # assuming depth is stored in mm
            depths.append(depth_in_meters)
        # Load semantic images
        semantic_path = Path(episode["episode_path"]) / f"semantic"
        semantic_files = sorted(list(semantic_path.glob("*.png")))
        semantic_files = [f for f in semantic_files if 'binary' not in f.name]
        semantic_meta_files = sorted(list(semantic_path.glob("*.json")))
        # load all semantic images using PIL
        semantics = [np.array(Image.open(str(f)).convert("RGB")) for f in semantic_files]
        # Load Poses
        pose_path = Path(episode["episode_path"]) / f"pose"
        pose_files = sorted(list(pose_path.glob("*.npy")))
        poses = [self._convert_pose(np.load(str(f))) for f in pose_files]
        assert len(rgbs) == len(depths) == len(poses), "Mismatch in number of frames"

        observations = []
        for i in range(len(rgbs)):
            obs = {
                "rgb": rgbs[i],
                "depth": depths[i],
                "pose": poses[i],
                "semantic": semantics[i],
            }
            observations.append(obs)
        self.observations = observations

    def _convert_pose(self, pose: np.ndarray) -> np.ndarray:
        """Convert pose."""
        pose = np.linalg.inv(pose)
        conversion = np.diag([1, -1, -1, 1])
        pose = conversion @ pose
        pose = np.linalg.inv(pose)
        return pose

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
    
    def _calculate_gt_metrics(self) -> Dict[str, Any]:
        """Calculate ground truth metrics for the current step."""
        resolution = self.occupancy_resolution
        obstacle_height_thresh = self.occupancy_obstacle_height_thresh
        obs_occupancy = OccupancyMap(resolution, obstacle_height_thresh)

        initial_pose = self.imagination_poses[0]
        for (depth, rgb, pose) in zip(self.imagination_key_depths, self.imagination_key_rgbs, self.imagination_key_poses):
            pose = np.linalg.inv(initial_pose) @ pose
            pcd = OracleDepthModel.depth_to_pcd(
                depth, self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy,
                rgb=rgb, T_cam2world=pose, invalid_val=0
            )

            # pcd = pcd.transform(np.linalg.inv(initial_pose))

            # position = pose[:3, 3]
            # rotation = pose[:3, :3]

            obs_occupancy.integrate(np.array(pcd.points), np.array([0,0,0]), np.eye(3), intrinsics=self.camera.intrinsics)

        # ## DEBUG
        # obs_occupancy.save_occupancy_map("/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/outputs/debug/obs_occupancy.png")
        ret = {
            "gt_occupancy": obs_occupancy,
        }

        return ret

    def _calculate_rendered_metrics(self, assets: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate rendered metrics for the current step."""
        colored_pcd = assets["imagine_pcd"]
        resolution = self.occupancy_resolution
        obstacle_height_thresh = self.occupancy_obstacle_height_thresh
        belief_occupancy = OccupancyMap(resolution, obstacle_height_thresh)
        belief_occupancy.integrate(np.array(colored_pcd.points), np.array([0,0,0]), np.eye(3), intrinsics=self.camera.intrinsics)
        rgb = assets["imagine_rgb"][-1]
        
        # ## DEBUG
        # belief_occupancy.save_occupancy_map("/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/wm_baselines/outputs/debug/belief_occupancy.png")
        ret = {
            "belief_occupancy": belief_occupancy,
        }

        return ret
