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
from belief_baselines.task_manager.base_task_manager import BaseTaskManager
from belief_baselines.agent.perception.metrics import Box3D, box3d_from_aabb
from belief_baselines.agent.perception.camera import Camera
from belief_baselines.utils.planning_utils import rotation_angle
from belief_baselines.agent.vlm.vlm import VLM
from belief_baselines.agent.vlm.object_detection import segment_label_with_gemini


class SpocObjCompletionTaskManager(BaseTaskManager):
    def __init__(self, embodied_config: DictConfig, stretch_controller: StretchController, **kwargs):
        super().__init__(embodied_config)
        self.episode_root = embodied_config.episode_root
        self.adjacent_angle = embodied_config.adjacent_angle
        self.adjacent_distance = embodied_config.adjacent_distance
        self.num_steps = None  # to be set during reset
        self.stretch_controller = stretch_controller

        self.episode_list = sorted([p for p in Path(self.episode_root).glob("*/") if p.is_dir()])
        self.episodes = [self._load_episode(p) for p in self.episode_list]
        self.positions = None
        self.rotations = None
        self.visibility_percents = None
        self.observations = None
        self.imagination_poses = None
        self.done = False
        self.current_episode_index = -1
        self.current_target_obj = None
        self.current_house_index = None
    
    @property
    def current_step(self):
        return self._current_step
    
    @property
    def current_ep_name(self):
        return f"{self.current_episode_index}_{self.episodes[self.current_episode_index]['house_index']}_{self.episodes[self.current_episode_index]['target_object']}"

    def set_camera(self, camera: Camera):
        self.camera = camera
    
    def set_vlm(self, vlm: VLM):
        self.vlm = vlm

    def get_observation(self):
        """Get the current observation of the episode."""
        obs = self.observations[self._current_step]
        imagination_poses = [self.observations[i]['pose'] for i in range(self._current_step, len(self.observations))]
        # imagination_key_poses = self._extract_key_poses(imagination_poses)
        self.imagination_poses = imagination_poses
        visibility = self.visibility_percents[self._current_step]
        if visibility == 1.0:
            self.done = True
        self._current_step += 1
        return obs

    def get_imagination_poses(self):
        """Get the imagination poses for the current step."""
        return self.imagination_poses
    
    def _extract_key_poses(self, poses):
        key_poses = []
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
                z_previous = z_idx
                t_previous = t_idx
        return key_poses

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
        self.positions = episode["positions"]
        self.rotations = episode["rotations"]
        self.visibility_percents = episode["visibility_percent"]
        self.num_steps = episode["oracle_length"]
        self.done = False
        self.imagination_poses = []
        self._load_observation()

        print(f"Starting episode {self.current_episode_index}: House {self.current_house_index}, Target Object: {self.current_target_obj}")
        house_data = self._load_house_from_prior(self.current_house_index)
        self.stretch_controller.reset(house_data)
        self.stretch_controller.step(
            action="TeleportFull",
            position=self.positions[0],
            rotation=self.rotations[0],
            horizon=0.0,  # Level camera view
            standing=True  # Agent is standing
        )
        # scene_graph = self.stretch_controller.current_scene_json['objects']

        super().reset(idx)

    def is_done(self):
        return self.done

    def get_final_log(self, metrics: dict = None):
        episode = self.episodes[self.current_episode_index]
        final_log = {
            "episode_index": self.current_episode_index,
            "house_index": self.current_house_index,
            "target_object": self.current_target_obj,
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

    def _load_episode(self, episode_path: Path):
        """Load episode info (episode_index, house_id, target_object) 
        from the given path."""
        metadata_path = episode_path / "trajectory_metadata.json"
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        if "," in metadata["object_id"]:
            metadata["object_id"] = metadata["object_id"].split(",")[0]  # take the first object if multiple
        episode = {
            "episode_path": str(episode_path),
            "house_index": int(metadata["house_id"].split("_")[-1]),
            "target_object": metadata["object_id"],
            "oracle_length": metadata["frames"],
            "positions": metadata["positions"][::-1],
            "rotations": metadata["rotations"][::-1],
            "visibility_percent": metadata["visibility_percent"][::-1],
        }
        return episode
    
    def _load_observation(self, depth_scale: float = 1000.0):
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
        # load all depth images using PIL
        depths = [np.array(Image.open(str(f))) / depth_scale for f in depth_files]
        # Load semantic images
        semantic_path = Path(episode["episode_path"]) / f"semantic"
        semantic_files = sorted(list(semantic_path.glob("*.png")))
        semantic_files = [f for f in semantic_files if 'binary' not in f.name]
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
        observations.reverse()
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
        position = self.positions[-1]
        rotation = self.rotations[-1]
        self.stretch_controller.step(
            action="TeleportFull",
            position=position,
            rotation=rotation,
            horizon=0.0,  # Level camera view
            standing=True  # Agent is standing
        )
        scene_graph = self.stretch_controller.current_scene_json['objects']
        current_target_info = find_object_node(scene_graph, self.current_target_obj)
        color = self.stretch_controller.controller.last_event.object_id_to_color.get(current_target_info['id'], None)
        semantic = self.observations[-1]['semantic']
        # get binary mask of the target object based on color
        target_mask = np.all(np.array(semantic) == np.array(color), axis=-1)
        # calculate 2D bounding box
        ys, xs = np.where(target_mask)
        if len(xs) == 0 or len(ys) == 0:
            print("Target object not visible in the current observation.")
            return {}
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        bbox_2d = [int(x_min), int(y_min), int(x_max), int(y_max)]
        # find a better seed point as the closest point to the center of the bbox
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        distances = (xs - center_x) ** 2 + (ys - center_y) ** 2
        min_idx = np.argmin(distances)
        seed_x = xs[min_idx]
        seed_y = ys[min_idx]
        seed = (seed_y, seed_x)
        # normalize seed to [0,1]
        h, w = semantic.shape[:2]
        seed_norm = (seed_x / w, seed_y / h)
        bbox_image = np.array(self.observations[-1]['rgb']).copy()
        bbox_image = bbox_image.astype(np.uint8)
        # crop the bbox region
        bbox_crop = bbox_image[y_min:y_max, x_min:x_max]
        # # DEBUG: save bbox image
        # cv2.imwrite("/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/reasoning/bbox_image.png", bbox_image)
        # cv2.imwrite("/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/reasoning/bbox_image_crop.png", bbox_crop)
        # calculate 3D point cloud of the target object
        rgb = np.array(self.observations[-1]['rgb'])
        depth = np.array(self.observations[-1]['depth'])
        assert self.camera is not None, "Camera not set in TaskManager."
        pose = self.observations[-1]['pose']
        target_pcd = self._masked_depth_to_world_pcd(
                                                depth,
                                                target_mask, 
                                                self.camera.intrinsics,
                                                T_world_cam=pose,
                                                rgb=rgb
                                            )

        ctr = target_pcd.get_center()  # centroid of all points (mean)
        print("Center of GT target PCD:", ctr)
        target_pcd.translate(-ctr)
        # extract target 3D bounding box from target pcd
        box_aabb = box3d_from_aabb(target_pcd)
        
        # DEBUG: save target_pcd
        # o3d.io.write_point_cloud("/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/reasoning/target_pcd.ply", target_pcd)
        ret = {
            "gt_bbox_2d": bbox_2d,
            "gt_bbox_image": bbox_crop,
            "gt_target_pcd": target_pcd,
            "gt_bbox_3d": box_aabb,
            "target_seed": seed_norm,
            "gt_w": w,
            "gt_h": h,
        }
        return ret

    def _calculate_rendered_metrics(self, assets: Dict[str, Any], seed: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate rendered metrics for the current step."""
        imagined_rgb = assets["imagine_rgb"][-1]
        imagined_depth = assets["imagine_depth"][-1]
        rgb = np.array(imagined_rgb)
        depth = np.array(imagined_depth)[0]
        target_obj = self.fix_object_name(self.current_target_obj)
        detected = False
        # object detection
        detections, annotated = segment_label_with_gemini(image_np=rgb, label=target_obj, return_annotated=True)
        if len(detections) == 0 or detections.mask is None:
            print(f"Object '{target_obj}' NOT detected by VLM.")
            ret = {
                "rendered_bbox_2d": None,
                "rendered_bbox_image": None,
                "rendered_target_pcd": o3d.geometry.PointCloud(),
                "rendered_bbox_3d": None,
            }
            if self.vlm is not None:
                ret["vlm_obj_recognition"] = 0
            return ret
        box = detections.xyxy[0]
        x_min, y_min, x_max, y_max = box.tolist()
        target_mask = detections.mask[0]
        detected = True
        print(f"Object '{target_obj}' detected by VLM.")
        
        x_min, x_max = int(x_min), int(x_max)
        y_min, y_max = int(y_min), int(y_max)
        bbox_2d = [x_min, y_min, x_max, y_max]
        bbox_image = np.array(imagined_rgb).copy()
        bbox_image = bbox_image.astype(np.uint8)
        gt_bbox = assets.get("gt_bbox_2d", None)
        if gt_bbox is not None:
            gt_x_min, gt_y_min, gt_x_max, gt_y_max = gt_bbox
            gt_h = assets["gt_h"]
            gt_w = assets["gt_w"]
            x_min = int(gt_x_min / gt_w * rgb.shape[1])
            x_max = int(gt_x_max / gt_w * rgb.shape[1])
            y_min = int(gt_y_min / gt_h * rgb.shape[0])
            y_max = int(gt_y_max / gt_h * rgb.shape[0])
            bbox_crop = bbox_image[y_min:y_max, x_min:x_max]
            # reshape to gt size
            bbox_crop = cv2.resize(bbox_crop, ((gt_x_max - gt_x_min), (gt_y_max - gt_y_min)))
            print("Rendered bbox (blue) vs GT bbox (green)")
        # # DEBUG: save bbox image
        # cv2.imwrite("/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/reasoning/imagined_bbox_image.png", bbox_image)
        # cv2.imwrite("/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/reasoning/imagined_bbox_image_crop.png", bbox_crop)
        # calculate 3D point cloud of the target object
        assert self.camera is not None, "Camera not set in TaskManager."
        pose = self.observations[-1]['pose']
        target_pcd = self._masked_depth_to_world_pcd(depth, target_mask, self.camera.intrinsics, T_world_cam=pose, rgb=rgb)
        # # save target_pcd
        # o3d.io.write_point_cloud("/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/reasoning/imagined_target_pcd.ply", target_pcd)
        # extract target 3D bounding box from target pcd
        ctr = target_pcd.get_center()  # centroid of all points (mean)
        print("Center of rendered target PCD:", ctr)
        target_pcd.translate(-ctr)
        box_aabb = box3d_from_aabb(target_pcd)
        # vlm object recognition
        if self.vlm is not None:
            response = self.vlm.prompt_score_obj_image(rgb, target_obj)
            vlm_obj_recognition = int(response['parsed'])
        ret = {
            "rendered_bbox_2d": bbox_2d,
            "rendered_bbox_image": bbox_crop,
            "rendered_target_pcd": target_pcd,
            "rendered_bbox_3d": box_aabb,
        }
        if self.vlm is not None:
            ret["vlm_obj_recognition"] = vlm_obj_recognition

        return ret

    def _masked_depth_to_world_pcd(
        self,
        depth: np.ndarray,                  # (H, W) depth map (same units as you want in output, e.g., meters)
        mask: np.ndarray,                   # (H, W) bool or {0,1}, True where you want points
        K: np.ndarray,                      # (3, 3) intrinsics [[fx, 0, cx],[0, fy, cy],[0,0,1]]
        T_world_cam: np.ndarray,            # (4, 4) camera-to-world pose (R|t) of the *camera* w.r.t. world
        *,
        rgb: np.ndarray = None,             # optional (H, W, 3) uint8 RGB aligned with depth
        depth_min: float = 1e-6,            # filter invalid/zero/too-small depths
        depth_max: float = np.inf,          # optional max range filter
        return_o3d: bool = True             # return open3d.geometry.PointCloud if True
    ):
        """
        Returns:
            points_world: (N, 3) float32 world-frame points
            (optional) colors: (N, 3) uint8 if rgb is given
            If return_o3d=True: returns an open3d.geometry.PointCloud (with colors if rgb provided)
        Assumptions:
            - depth[i, j] is the Z in the *camera* frame at pixel (row=i, col=j).
            - T_world_cam maps camera-frame coords to world: X_w = R_wc @ X_c + t_wc
        """
        H, W = depth.shape
        if mask.dtype != bool:
            mask = mask.astype(bool)

        # Validity mask
        valid = mask & np.isfinite(depth) & (depth > depth_min) & (depth < depth_max)
        if not np.any(valid):
            if return_o3d:
                return o3d.geometry.PointCloud()
            return (np.zeros((0, 3), dtype=np.float32),) if rgb is None else (np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8))

        v_idx, u_idx = np.where(valid)  # rows (v), cols (u)
        z = depth[v_idx, u_idx].astype(np.float32)

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # Back-project to camera frame
        x = (u_idx.astype(np.float32) - cx) * z / fx
        y = (v_idx.astype(np.float32) - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)  # (N, 3)

        # Transform to world frame: X_w = R_wc * X_c + t_wc
        R_wc = T_world_cam[:3, :3].astype(np.float32)
        t_wc = T_world_cam[:3, 3].astype(np.float32)
        points_world = (pts_cam @ R_wc.T) + t_wc  # (N, 3)

        if return_o3d:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_world.astype(np.float64))
            if rgb is not None:
                colors = rgb[v_idx, u_idx].astype(np.float32) / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
            return pcd

        if rgb is not None:
            colors = rgb[v_idx, u_idx]  # (N, 3) uint8
            return points_world.astype(np.float32), colors
        else:
            return points_world.astype(np.float32)

    def _extract_mask_seeded(
        self,
        I: np.ndarray,
        seed_rc: tuple[int, int],
        *,
        p_thresh: float = 95.0,          # percentile for "high enough" (bright mode)
        object_polarity: str = "bright", # "bright" | "dark" | "auto"
        smooth_sigma: float = 1.0,
        connectivity: int = 2,           # 2 => 8-connectivity
        r_connect: int = 2,              # radius (px) to fuse nearby high islands for labeling only
        open_rad: int = 0,               # optional speckle removal before dilation
        close_rad: int = 0,              # optional gap-bridging inside each island (rarely needed here)
        min_area: int = 10,              # area filter on *true-high* pixels
        max_area_frac: float = 0.6,      # cap runaway blobs
    ):
        """
        Single-threshold 'superlevel set' with connectivity dilation:
        - Threshold once
        - Dilate for connectivity (labeling), but measure/return only true-high pixels.

        Ranking priority: intensity -> area -> distance_to_seed (smaller is better).

        Returns: mask (H,W) bool, info dict
        """
        I = np.asarray(I)
        H, W = I.shape
        r0, c0 = seed_rc
        assert 0 <= r0 < H and 0 <= c0 < W

        # 1) Denoise & normalize
        I_s = gaussian(I.astype(np.float32), sigma=smooth_sigma, preserve_range=True)
        vmin, vmax = np.nanpercentile(I_s, [0.5, 99.5])
        I_n = np.clip((I_s - vmin) / max(vmax - vmin, 1e-6), 0, 1)

        # 2) Polarity + single threshold
        if object_polarity == "auto":
            hw = 20
            r1, r2 = max(0, r0 - hw), min(H, r0 + hw + 1)
            c1, c2 = max(0, c0 - hw), min(W, c0 + hw + 1)
            loc_med = np.median(I_n[r1:r2, c1:c2])
            object_polarity = "bright" if I_n[r0, c0] >= loc_med else "dark"

        if object_polarity == "bright":
            T = np.percentile(I_n, p_thresh)
            high = I_n >= T
        else:
            T = np.percentile(I_n, 100.0 - p_thresh)
            high = I_n <= T

        # Optional light cleaning on true-high pixels (before connectivity dilation)
        if open_rad > 0:
            high = binary_opening(high, footprint=disk(open_rad))
        if close_rad > 0:
            high = binary_closing(high, footprint=disk(close_rad))

        # 3) Connectivity dilation (for labeling only)
        label_support = binary_dilation(high, footprint=disk(r_connect)) if r_connect > 0 else high

        if not label_support.any():
            return np.zeros_like(high), {
                "mode": "single_thresh",
                "polarity": object_polarity, "threshold": float(T),
                "reason": "no pixels above threshold"
            }

        # 4) Label on the dilated support; measure on true-high
        lab = label(label_support, connectivity=connectivity)

        def seed_distance(comp_mask: np.ndarray) -> float:
            # comp_mask is the true-high pixels restricted to a labeled region
            if comp_mask[r0, c0]:
                return 0.0
            rr, cc = np.nonzero(comp_mask)
            if rr.size == 0:
                return np.inf
            d2 = (rr - r0) * (rr - r0) + (cc - c0) * (cc - c0)
            return float(np.sqrt(d2.min()))

        best = dict(k=None, area=-1, dist=np.inf, inten=-np.inf, score=(-np.inf, -1, -np.inf))
        for k in range(1, lab.max() + 1):
            region_mask = (lab == k)
            comp_true_high = region_mask & high  # restrict to true-high pixels only

            A = int(comp_true_high.sum())
            if A < max(1, min_area):
                continue
            if max_area_frac < 1.0 and A > int(max_area_frac * H * W):
                # skip obviously huge regions
                continue

            dist = seed_distance(comp_true_high)
            vals = I_n[comp_true_high]
            inten = float(np.median(vals)) if vals.size else -np.inf
            if not np.isfinite(inten):
                inten = -np.inf

            # New ranking: intensity (desc), then area (desc), then distance (asc)
            # Use -dist to convert to a "larger is better" tie-breaker.
            score = (inten, A, -dist)

            if score > best["score"]:
                best.update(k=k, area=A, dist=dist, inten=inten, score=score)

        if best["k"] is None:
            return np.zeros_like(high), {
                "mode": "single_thresh",
                "polarity": object_polarity, "threshold": float(T),
                "reason": "no component after area filter"
            }

        mask = (lab == best["k"]) & high  # final mask = true-high pixels of the chosen region

        info = {
            "mode": "single_thresh_connectivity_dilation",
            "polarity": object_polarity,
            "threshold": float(T),
            "area": int(best["area"]),
            "distance_px": float(best["dist"]),
            "median_intensity": float(best["inten"]),
            "seed_inside": bool(mask[r0, c0]),
            "r_connect": int(r_connect),
            "p_thresh": float(p_thresh),
            "rank_order": "intensity > area > -distance",
        }
        return mask, info