from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig
import numpy as np
import open3d as o3d
import imageio
import os
import torch
import copy
from pathlib import Path
from torch import Tensor
import torch.nn.functional as F
import math
from splat_belief.splat.types import Gaussians
from splat_belief.splat.ply_export import export_gaussians_to_ply
try:
    from general_utils import to_gpu, normalize_to_neg_one_to_one
except ImportError:
    from splat_belief.utils.vision_utils import to_gpu, normalize_to_neg_one_to_one
try:
    from rollout_utils import prepare_video
except ImportError:
    def prepare_video(frames, depths, semantics):
        """Fallback: convert tensor frames to numpy uint8, pass depths/semantics through."""
        out_frames = []
        for f in frames:
            if hasattr(f, 'cpu'):
                f = f.cpu().detach().numpy()
            if f.max() <= 1.0:
                f = (f * 255).clip(0, 255).astype(np.uint8)
            else:
                f = f.clip(0, 255).astype(np.uint8)
            out_frames.append(f)
        return out_frames, depths, semantics
from splat_belief.diffusion.diffusion_temporal import Trainer as ModelWrapper
from wm_baselines.utils.common_utils import with_timing
from wm_baselines.utils.data_classes import Frame, RenderOutput
from einops import rearrange
from wm_baselines.agent.perception.camera import Camera
from wm_baselines.agent.perception.occupancy import OccupancyMap
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.utils.vision_utils import (
    pose_lh2rh, pose_gl_world2cam_to_open3d_world2cam, points_gl_to_open3d, plot_two_poses, flip_yaw_in_Twc
)

class Belief3DModel(BaseWorldModel):
    """3D belief world model maintaining both observation and belief occupancy maps.

    Responsibilities:
    - hold config reference
    - maintain observation and belief occupancy maps
    - implement update_observation() to update the occupancy maps
    - implement render_image() and render_video() to visualize the occupancy maps
    """

    def __init__(
        self, 
        model: ModelWrapper,
        obs_occupancy: OccupancyMap, 
        belief_occupancy: OccupancyMap, 
        camera: Camera,
        adjacent_angle: float,
        adjacent_distance: float,
        coverage_threshold: float = 0.5,
        fast_sampling: bool = True,
        obs_filter_border_gaussians: bool = False,
        obs_depth_min: float = 0.1,
        obs_depth_max: float = 10.0,
        disable_imagination: bool = False,
        occupancy_min_opacity: float = 0.1,
        occupancy_max_scale_m: float = 0.3,
        occupancy_max_range_m: float = 4.0,
        occupancy_border_px: int = 2,
        occlusion_filter_enabled: bool = True,
        occlusion_target_buffer_cells: int = 3,
        inc_pcd_flip_y: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.model.model.eval()
        self.obs_occupancy = obs_occupancy
        self.belief_occupancy = belief_occupancy
        self.obs_scene: Gaussians = None
        self.inc_scene: Gaussians = None
        self.belief_scene: Gaussians = None
        self.current_rgb: Tensor = None
        self.current_pose: Tensor = None
        self.normalize = normalize_to_neg_one_to_one
        self.model_image_size = self.model.image_size
        self.camera = camera
        self.initial_location = {}
        self.resample_pcd = True
        self.resample_pcd_check_dict = {}
        self.adjacent_angle = adjacent_angle
        self.adjacent_distance = adjacent_distance
        self.fast_sampling = fast_sampling
        self.disable_imagination = disable_imagination
        # resample a new observation only if the new cone of view reveals at least this
        # fraction of previously unknown cells within that cone
        self.resample_unknown_coverage_threshold = coverage_threshold
        self.step = -1
        self.state_step = 0
        self._metrics = {
            "model_inference_time_obs": 0.0,
            "model_inference_time_imagine": 0.0,
            "update_occupancy_time": 0.0,
            "obs_steps": -1,
        }
        self.imagination_model = None
        self.obs_filter_border_gaussians = obs_filter_border_gaussians
        self.obs_depth_min = obs_depth_min
        self.obs_depth_max = obs_depth_max
        # inc_scene -> occupancy filtering knobs
        self.occupancy_min_opacity = float(occupancy_min_opacity)
        self.occupancy_max_scale_m = float(occupancy_max_scale_m)
        self.occupancy_max_range_m = float(occupancy_max_range_m)
        self.occupancy_border_px = int(occupancy_border_px)
        # When True, drop incremental gaussians whose ray from the sensor crosses an
        # already-known obstacle cell in self.obs_occupancy (kills through-wall imagination).
        self.occlusion_filter_enabled = bool(occlusion_filter_enabled)
        # Stop the ray-occlusion check this many cells before the target so we don't
        # reject neighbour points on the same wall (curtain self-rejection).
        self.occlusion_target_buffer_cells = int(occlusion_target_buffer_cells)
        # If True, negate Y of the incremental pcd before handing it to
        # OccupancyMap (legacy behavior). Set False to test whether the
        # ceiling/obstacle thresholds were tuned against the *unflipped* frame.
        self.inc_pcd_flip_y = bool(inc_pcd_flip_y)
    
    @property
    def augmented_scene(self) -> Gaussians:
        if self.obs_scene is not None and self.belief_scene is not None:
            return self.obs_scene + self.belief_scene
        elif self.obs_scene is not None:
            return self.obs_scene
        elif self.belief_scene is not None:
            return self.belief_scene
        else:
            return None

    def reset(self):
        """Reset the occupancy maps."""
        self.obs_occupancy = OccupancyMap(
            resolution=self.obs_occupancy.resolution,
            obstacle_height_thresh=self.obs_occupancy.obstacle_height_thresh,
            ceiling_height=self.obs_occupancy.ceiling_height,
            max_range=self.obs_occupancy.max_range,
            free_overrides_occupied=self.obs_occupancy.free_overrides_occupied,
            seed=self._seed,
            agent_free_radius=self.obs_occupancy.agent_free_radius,
            flip_y=self.obs_occupancy.flip_y,
            debug=self.obs_occupancy.debug,
        )
        self.belief_occupancy = OccupancyMap(
            resolution=self.belief_occupancy.resolution,
            obstacle_height_thresh=self.belief_occupancy.obstacle_height_thresh,
            ceiling_height=self.belief_occupancy.ceiling_height,
            max_range=self.belief_occupancy.max_range,
            free_overrides_occupied=self.belief_occupancy.free_overrides_occupied,
            seed=self._seed,
            agent_free_radius=self.belief_occupancy.agent_free_radius,
            flip_y=self.belief_occupancy.flip_y,
            debug=self.belief_occupancy.debug,
        )

        self.model.model.model.reset_timestep()
        self.model.ema.ema_model.model.reset_timestep()

        self.obs_scene = None
        self.inc_scene = None
        self.belief_scene = None
        self.current_rgb = None
        self.current_pose = None

        self.initial_location = {}
        self.resample_pcd = True
        self.resample_pcd_check_dict = {}
        self.step = -1
        self.state_step = 0
        self._metrics = {
            "model_inference_time_obs": 0.0,
            "model_inference_time_imagine": 0.0,
            "update_occupancy_time": 0.0,
            "obs_steps": -1,
        }
        self.imagination_model = None

    def update_observation(self, observation: Dict[str, Any], force_update: bool = False) -> Dict[str, Any]:
        """Update the world model with a new observation."""
        self.step += 1
        self._metrics["obs_steps"] += 1
        rgb = observation.get("rgb")
        pose = observation.get("pose")  # (4, 4) world to camera
        if 'depth' in observation:
            depth = observation.get("depth")
        else:
            depth = None

        if self.initial_location.get('pose') is None:
            self.initial_location['pose'] = pose
        
        if self.initial_location.get('pose') is not None:
            pose_map = np.linalg.inv(self.initial_location['pose']) @ pose

        self._update_resample_pcd(pose_map, force=force_update)

        pcd = None
        if self.resample_pcd:
            self.resample_pcd = False
            # pose_map_converted = pose_gl_world2cam_to_open3d_world2cam(pose_map)
            pose_map_converted = pose_map
            pose_map_tensor = torch.tensor(pose_map_converted, dtype=torch.float32)
            R = pose_map_tensor[:3, :3]
            t = pose_map_tensor[:3, 3]
            angle = np.radians(45)
            R_y = torch.tensor([
                [np.cos(angle), 0, np.sin(angle)],
                [0, 1, 0],
                [-np.sin(angle), 0, np.cos(angle)]
            ], dtype=torch.float32)
            imagine_R = R @ R_y
            imagine_pose = pose_map_tensor.clone()
            imagine_pose[:3, :3] = imagine_R
            imagine_pose = imagine_pose.unsqueeze(0)
            # prepare obs
            rgb_tensor = (
                torch.tensor(
                    rgb.astype(np.float32)
                ).permute(2, 0, 1)
                / 255.0
            )
            rgb_tensor = F.interpolate(
                rgb_tensor.unsqueeze(0),
                size=(self.model_image_size, self.model_image_size),
                mode="bilinear",
                antialias=True,
            )[0]
            rgb_normalized = self.normalize(rgb_tensor).unsqueeze(0) # normalize to [-1, 1]
            obs_pose = pose_map_tensor.unsqueeze(0)
            self.current_rgb = rgb_normalized.squeeze(0)
            self.current_pose = obs_pose.squeeze(0)
            obs_pose = torch.tensor(obs_pose, dtype=torch.float32)
            imagine_pose = torch.tensor(imagine_pose, dtype=torch.float32)
            _, exe_time = self._update_belief(rgb_normalized, obs_pose, imagine_pose, depth=depth)
            self._metrics["model_inference_time_obs"] += exe_time
            position = pose_map[:3, 3]
            rotation = pose_map[:3, :3]
            inc_pcd = self._extract_inc_pcd(position=position, rotation=rotation)
            if inc_pcd.shape[0] > 0:
                _, exe_time = self.obs_occupancy.integrate(
                    inc_pcd, position, rotation, intrinsics=self.camera.intrinsics,
                )
                self._metrics["update_occupancy_time"] += exe_time
            pcd = self._extract_obs_colored_pcd()
        
        assets = {
            "rgb": rgb,
            "occupancy": self.obs_occupancy,
            "pose": pose_map,
            "initial_pose": self.initial_location['pose'],
            "pcd": pcd if pcd is not None else None,
        }
        return assets

    def render_goal_images(
        self,
        goals: List[np.ndarray],           # List of (3,) camera positions
        forwards: List[np.ndarray],        # List of (3,) forward vectors
        initial_pose: Union[Tensor, np.ndarray],  # (4,4) world-from-root
        intrinsics: Optional[Any] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        query_label: Optional[str] = None,
    ) -> List[Any]:
        """Render images of the model given a list of goal camera views."""
        assert len(goals) == len(forwards), "Goals and forwards must have the same length"
        # ensure float
        images = []
        semantics = []
        depths = []
        poses = []
        for goal, forward in zip(goals, forwards):
            goal = np.asarray(goal, dtype=np.float64).reshape(3)
            forward = BaseWorldModel.normalize(np.asarray(forward, dtype=np.float64).reshape(3))

            # Camera Z looks along the forward direction
            z_axis = BaseWorldModel.normalize(forward)
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = BaseWorldModel.normalize(np.cross(y_axis, z_axis))

            R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)   # (3,3)
            t = goal.astype(np.float64).reshape(3, 1)

            T_cam2world = np.eye(4, dtype=np.float64)
            T_cam2world[:3, :3] = R
            T_cam2world[:3, 3:4] = t

            # T_cam2world = pose_gl_world2cam_to_open3d_world2cam(T_cam2world)
            (key_output, _, _), exe_time = self._imagine_in_place(T_cam2world, query_label=query_label)
            select = len(key_output.rgb) // 4
            images.append(key_output.rgb[select])  # (H, W, 3)
            if key_output.semantic is not None and len(key_output.semantic) > 0:
                semantics.append(key_output.semantic[select])
            if key_output.depth is not None and len(key_output.depth) > 0:
                depths.append(key_output.depth[select])
            if key_output.pose is not None and len(key_output.pose) > 0:
                poses.append(key_output.pose[select])
            self._metrics["model_inference_time_imagine"] += exe_time
        render_output = RenderOutput(rgb=images, semantic=semantics, depth=depths, pose=poses)

        return render_output

    def _update_resample_pcd(self, current_pose: np.ndarray, force: bool = False) -> None:
        """Decide whether to resample PCD based on new-view unknown coverage."""
        # Always take the very first frame or forced updates
        if force or self.step == 0:
            self.resample_pcd = True
            self.resample_pcd_check_dict["pose_previous"] = current_pose.copy()
            return

        # If we don't yet have an occupancy map, we must integrate
        if self.obs_occupancy.occupancy is None:
            self.resample_pcd = True
            self.resample_pcd_check_dict["pose_previous"] = current_pose.copy()
            return

        occ = self.obs_occupancy

        # Derive yaw (world frame, looking along +Z after conversion) similar to OccupancyMap.integrate
        R = current_pose[:3, :3]
        forward = R @ np.array([0.0, 0.0, 1.0])
        yaw = -math.atan2(forward[0], forward[2])
        yaw = yaw % (2 * math.pi)
        position = tuple(current_pose[:3, 3])

        # Build a visibility mask for the new cone of view (without mutating the map)
        res = float(occ.resolution)
        r_max = int(math.ceil(occ.max_range / res))
        intrinsics = self.camera.intrinsics

        # horizontal FOV from intrinsics (if provided)
        if intrinsics is not None:
            fx = float(intrinsics[0, 0])
            cx = float(intrinsics[0, 2])
            w = int(round(2 * cx))
            fov_rad = 2.0 * math.atan(w / (2.0 * fx))
        else:
            fov_rad = None

        if fov_rad is None:
            N = max(int(math.ceil(2.0 * math.pi * occ.max_range / res)), 360)
            angles = np.linspace(0.0, 2.0 * math.pi, N, endpoint=False)
        else:
            N = max(int(math.ceil(fov_rad * occ.max_range / res)), 30)
            angles = np.linspace(yaw - fov_rad / 2.0, yaw + fov_rad / 2.0, N, endpoint=False)

        angles = angles + math.pi / 2.0  # rotate so 0 rad points +Z (X-right, Z-forward)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        ds = np.arange(1, r_max + 1) * res

        shifts = [(0.0, 0.0)]

        def raycast_free_mask(origin_x: float, origin_z: float) -> np.ndarray:
            Xw = origin_x + np.outer(ds, cos_a)
            Zw = origin_z + np.outer(ds, sin_a)
            cols = np.clip(((Xw - occ.x_min) / res).astype(int), 0, occ.nx - 1)
            rows = np.clip(((Zw - occ.z_min) / res).astype(int), 0, occ.nz - 1)
            occ_ray = occ.occupancy[rows, cols]
            blocked = (occ_ray == 1)
            any_blocked = blocked.any(axis=0)
            first_obs = np.where(any_blocked, blocked.argmax(axis=0), r_max)
            free_mask = np.zeros_like(occ.occupancy, dtype=bool)
            for i in range(angles.shape[0]):
                end = int(first_obs[i])
                if end > 0:
                    rr = rows[:end, i]
                    cc = cols[:end, i]
                    free_mask[rr, cc] = True
            return free_mask

        all_free_mask: Optional[np.ndarray] = None
        base_x, _, base_z = float(position[0]), float(position[1]), float(position[2])
        for dx, dz in shifts:
            fm = raycast_free_mask(base_x + dx, base_z + dz)
            if all_free_mask is None:
                all_free_mask = fm
            else:
                all_free_mask &= fm

        if all_free_mask is None:
            self.resample_pcd = False
            return

        total_view_cells = int(all_free_mask.sum())
        if total_view_cells == 0:
            self.resample_pcd = False
            return

        unknown_in_view = int(((occ.occupancy == -1) & all_free_mask).sum())
        coverage_ratio = unknown_in_view / float(total_view_cells)

        self.resample_pcd = coverage_ratio >= self.resample_unknown_coverage_threshold
        self.resample_pcd_check_dict["pose_previous"] = current_pose.copy()

    @with_timing
    def _update_belief(self, rgb: Tensor, pose: Tensor, imagine_pose: Tensor, depth: Optional[Tensor]=None):
        """Update the belief occupancy map using the diffusion model."""
        # Build inference input
        inp = {
            "ctxt_c2w": pose,
            "ctxt_rgb": rgb,
            "trgt_c2w": imagine_pose,
            "intrinsics": torch.tensor(self.camera.intrinsics),
            "image_shape": torch.tensor([rgb.shape[-2], rgb.shape[-1]]),
            "near": torch.tensor(self.camera.near),
            "far": torch.tensor(self.camera.far),
        }
        inp = to_gpu(inp, "cuda")
        for k in inp.keys():
            inp[k] = inp[k].unsqueeze(0)
        # Save a deep copy of the model for imagination
        self.imagination_model = copy.deepcopy(self.model.ema.ema_model)
        # Run inference
        # NOTE: filter_border_gaussians is NOT passed to .sample() because that would zero out
        # border-pixel gaussians and degrade rendering quality. Border filtering for occupancy
        # ingestion is applied later in _extract_inc_pcd via self.obs_filter_border_gaussians.
        self.model.ema.ema_model.sample(batch_size=1, inp=inp, state_t=self.state_step, filter_border_gaussians=False, depth_inference_min=self.obs_depth_min, depth_inference_max=self.obs_depth_max)
        self.obs_scene = self.model.ema.ema_model.model.history_gaussians.clone()
        self.inc_scene = self.model.ema.ema_model.model.incremental_gaussians.clone()
        self.belief_scene = self.model.ema.ema_model.model.belief_gaussians.clone()
        self.state_step += 1

    def _pose_delta(self, pose_a: Union[Tensor, np.ndarray], pose_b: Union[Tensor, np.ndarray]) -> tuple[float, float]:
        """Return translation distance and forward-vector angle between two camera poses."""
        if not torch.is_tensor(pose_a):
            pose_a = torch.as_tensor(pose_a)
        if not torch.is_tensor(pose_b):
            pose_b = torch.as_tensor(pose_b)
        pose_a = pose_a.detach().float()
        pose_b = pose_b.detach().float()

        z_a = pose_a[:3, 2]
        z_b = pose_b[:3, 2]
        denom = torch.linalg.norm(z_a) * torch.linalg.norm(z_b)
        if float(denom.item()) <= 1e-9:
            angle = 0.0
        else:
            cos_angle = torch.clamp(torch.dot(z_a, z_b) / denom, -1.0, 1.0)
            angle = float(torch.arccos(cos_angle).item())
        distance = float(torch.linalg.norm(pose_b[:3, 3] - pose_a[:3, 3]).item())
        return distance, angle

    def _select_adjacent_bounded_key_frame_indices(
        self,
        render_poses: List[Union[Tensor, np.ndarray]],
    ) -> List[int]:
        """Select keyframes while keeping consecutive target deltas in-distribution.

        The old selector appended the first pose *after* a distance/angle threshold
        was crossed, so target pairs could be 1 raw step beyond the configured
        adjacent envelope. Here we commit the last pose before crossing whenever
        possible, and still include the final pose for metric/video coverage.
        """
        num_frames = len(render_poses)
        if num_frames == 0:
            return []
        if num_frames == 1:
            return [0]

        key_frame_indices: List[int] = []
        anchor_idx = 0
        idx = 1
        while idx < num_frames:
            distance, angle = self._pose_delta(render_poses[anchor_idx], render_poses[idx])
            exceeds_adjacent = (
                angle > self.adjacent_angle
                or distance > self.adjacent_distance
            )
            if exceeds_adjacent:
                candidate_idx = idx - 1
                # If a single raw adjacent step is already too large, keep the
                # current pose so the sequence still advances and remains complete.
                if candidate_idx <= anchor_idx:
                    candidate_idx = idx
                if not key_frame_indices or key_frame_indices[-1] != candidate_idx:
                    key_frame_indices.append(candidate_idx)
                anchor_idx = candidate_idx
                idx = anchor_idx + 1
                continue

            if idx == num_frames - 1 and idx != anchor_idx:
                key_frame_indices.append(idx)
            idx += 1

        return key_frame_indices
    
    @with_timing
    def _imagine_in_place(
        self, 
        imagine_goal_pose: Union[Tensor, np.ndarray], 
        return_full_video: bool=False, 
        return_belief_scene: bool=False, 
        num_key_frames: int=3,
        query_label: Optional[str]=None,
    ) -> Any:
        """Imagine in place given a goal pose, return key frame renders and optionally full video and belief scenes."""
        imagine_goal_pose = Tensor(imagine_goal_pose)
        imagine_model = copy.deepcopy(self.imagination_model)
        # interpolate imagine poses
        ctxt_c2w = self.current_pose[None, None, :, :]
        trgt_c2w = imagine_goal_pose[None, None, :, :]

        inp = {
            "ctxt_c2w": ctxt_c2w,
            "trgt_c2w": trgt_c2w,
            "near": torch.tensor(self.camera.near),
            "far": torch.tensor(self.camera.far),
        }
        render_poses = imagine_model.model.compute_poses("interpolation", inp, n=50)
        num_frames = len(render_poses)
        key_frame_indices = self._select_adjacent_bounded_key_frame_indices(render_poses)
        video_pose_indices = [0] + [idx for idx in key_frame_indices if idx != 0]
        video_poses = torch.stack(
            [render_poses[idx] for idx in video_pose_indices], 0
        )
        # select the first k key frames for imagination
        key_frame_indices = key_frame_indices[:num_key_frames]
        # run inference
        state_step = self.state_step - 1
        inp = {}
        inp["ctxt_c2w"] = torch.cat([self.current_pose.unsqueeze(0)], dim=0)
        inp["ctxt_rgb"] = torch.cat([self.current_rgb.unsqueeze(0)], dim=0)
        belief_scene =  []
        for imagine_t in range(len(key_frame_indices)):
            inp["trgt_c2w"] = render_poses[key_frame_indices[imagine_t]].unsqueeze(0)
            inp["intrinsics"] = torch.tensor(self.camera.intrinsics)
            inp["image_shape"] = torch.tensor([self.current_rgb.shape[-2], self.current_rgb.shape[-1]])
            inp["render_poses"] = video_poses
            inp["near"] = torch.tensor(self.camera.near)
            inp["far"] = torch.tensor(self.camera.far)
            if not imagine_t==len(key_frame_indices)-1:
                inp.pop("render_poses")
            inp = to_gpu(inp, "cuda")
            for k in inp.keys():
                if not k=="num_frames_render":
                    inp[k] = inp[k].unsqueeze(0)
            inp["num_frames_render"] = num_frames
            out = imagine_model.sample(batch_size=1, inp=inp, state_t=imagine_t+state_step, fast_sampling=self.fast_sampling)
            if return_belief_scene:
                belief_scene.append(imagine_model.model.belief_gaussians.clone())
            if not imagine_t==len(key_frame_indices)-1:
                inp["ctxt_c2w"] = render_poses[key_frame_indices[imagine_t]].unsqueeze(0)
                inp["ctxt_rgb"] = normalize_to_neg_one_to_one(out["images"])

        # double H and W for rendering
        h_render = self.camera.h * 2
        w_render = self.camera.w * 2

        if self.disable_imagination:
            gaussians = self.obs_scene
        else:
            gaussians = imagine_model.model.augmented_gaussians
        # render key frames
        key_frame_poses = [render_poses[idx] for idx in key_frame_indices]
        key_rgbs, key_depths, key_semantics = self.render_any_scene_video(gaussians, key_frame_poses, query_label=query_label, h=h_render, w=w_render)
        # prepare key frames
        key_rgbs, _, _ = prepare_video(key_rgbs, key_depths, None)
        # key_rgbs = prepare_video(key_rgbs)
        key_output = RenderOutput(rgb=key_rgbs, depth=key_depths, semantic=key_semantics, pose=key_frame_poses)

        # render full video
        full_output = None
        if return_full_video:
            full_frames, full_depths, full_semantics = self.render_any_scene_video(gaussians, render_poses, query_label=query_label, h=h_render, w=w_render)
            full_frames, _, _ = prepare_video(full_frames, full_depths, None)
            full_output = RenderOutput(rgb=full_frames, depth=full_depths, semantic=full_semantics, pose=render_poses)

        return key_output, full_output, belief_scene

    @with_timing
    def _imagine_along_path(
        self, 
        imagine_poses: List[Union[Tensor, np.ndarray]], 
        return_full_video: bool=False, 
        return_colored_pcd: bool=False, 
        one_middle_frame: bool=False,
        query_label: Optional[str]=None,
    ) -> Any:
        """Imagine in place given a goal pose, return key frame renders and optionally full video and belief scenes."""
        if one_middle_frame:
            mid_idx = len(imagine_poses) // 2
            imagine_poses = [imagine_poses[0], imagine_poses[mid_idx], imagine_poses[-1]]
        # take inverse to get world to camera
        imagine_poses = [np.linalg.inv(self.initial_location['pose']) @ pose for pose in imagine_poses]
        imagine_poses = [Tensor(pose) for pose in imagine_poses]
        imagine_model = copy.deepcopy(self.imagination_model)
        # interpolate imagine poses
        render_poses = imagine_poses
        num_frames = len(render_poses)
        key_frame_indices = self._select_adjacent_bounded_key_frame_indices(render_poses)
        video_pose_indices = [0] + [idx for idx in key_frame_indices if idx != 0]
        video_poses = torch.stack(
            [render_poses[idx] for idx in video_pose_indices], 0
        )
        # run inference
        state_step = self.state_step - 1
        inp = {}
        inp["ctxt_c2w"] = torch.cat([self.current_pose.unsqueeze(0)], dim=0)
        inp["ctxt_rgb"] = torch.cat([self.current_rgb.unsqueeze(0)], dim=0)
        for imagine_t in range(len(key_frame_indices)):
            inp["trgt_c2w"] = render_poses[key_frame_indices[imagine_t]].unsqueeze(0)
            inp["intrinsics"] = torch.tensor(self.camera.intrinsics)
            inp["image_shape"] = torch.tensor([self.current_rgb.shape[-2], self.current_rgb.shape[-1]])
            inp["render_poses"] = video_poses
            inp["near"] = torch.tensor(self.camera.near)
            inp["far"] = torch.tensor(self.camera.far)
            if not imagine_t==len(key_frame_indices)-1:
                inp.pop("render_poses")
            inp = to_gpu(inp, "cuda")
            for k in inp.keys():
                if not k=="num_frames_render":
                    inp[k] = inp[k].unsqueeze(0)
            inp["num_frames_render"] = num_frames
            out = imagine_model.sample(batch_size=1, inp=inp, state_t=imagine_t+state_step)
            if not imagine_t==len(key_frame_indices)-1:
                inp["ctxt_c2w"] = render_poses[key_frame_indices[imagine_t]].unsqueeze(0)
                inp["ctxt_rgb"] = normalize_to_neg_one_to_one(out["images"])

        if return_colored_pcd:
            colored_pcd = self._extract_colored_pcd(imagine_model.model.augmented_gaussians)
        else:
            colored_pcd = None

        # double H and W for rendering
        h_render = self.camera.h * 2
        w_render = self.camera.w * 2

        # render key frames
        key_frame_poses = [render_poses[idx] for idx in key_frame_indices]
        key_rgbs, key_depths, key_semantics = self.render_any_scene_video(
            imagine_model.model.augmented_gaussians, 
            key_frame_poses,
            query_label=query_label,
            h=h_render,
            w=w_render,
        )
        # prepare key frames
        key_rgbs, _, _ = prepare_video(key_rgbs, key_depths, None)
        # key_rgbs = prepare_video(key_rgbs)
        key_output = RenderOutput(rgb=key_rgbs, depth=key_depths, semantic=key_semantics, pose=key_frame_poses)

        # render full video
        full_output = None
        if return_full_video:
            full_frames, full_depths, full_semantics = self.render_any_scene_video(
                imagine_model.model.augmented_gaussians, 
                render_poses, 
                query_label=query_label,
                h=h_render,
                w=w_render,
            )
            full_frames, _, _ = prepare_video(full_frames, full_depths, None)
            full_output = RenderOutput(rgb=full_frames, depth=full_depths, semantic=full_semantics, pose=render_poses)

        return key_output, full_output, colored_pcd
    
    def _extract_inc_pcd(self, position=None, rotation=None):
        """Extract a denoised point cloud from `self.inc_scene` for occupancy ingestion.

        Filters applied (in order, all configurable via ctor):
          1. Opacity >= `occupancy_min_opacity` -- drops low-confidence diffusion gaussians.
          2. Largest axis length (sqrt of max covariance eigenvalue) <= `occupancy_max_scale_m`
             -- drops oversize "uncertainty blob" gaussians.
          3. If `position`/`rotation` provided: keep only points whose distance to the camera
             is in [obs_depth_min, occupancy_max_range_m] AND whose projection lies within the
             current camera frustum -- drops radial streaks behind/outside the viewing cone.
        """
        gaussians = self.inc_scene.float()
        means = gaussians.means.squeeze(0).detach()                # (N, 3) torch
        opacities = gaussians.opacities.squeeze(0).detach()        # (N,)   torch
        covariances = gaussians.covariances.squeeze(0).detach()    # (N,3,3) torch

        keep = opacities >= self.occupancy_min_opacity
        if keep.any():
            # Largest eigenvalue of 3x3 symmetric covariance == squared max-axis length.
            try:
                eig_max = torch.linalg.eigvalsh(covariances[keep])[..., -1]
                max_axis = torch.sqrt(eig_max.clamp_min(0))
                axis_keep = max_axis <= self.occupancy_max_scale_m
                # write back into the master mask
                idx = torch.nonzero(keep, as_tuple=True)[0]
                drop = idx[~axis_keep]
                keep[drop] = False
            except Exception:
                # eigvalsh failure shouldn't block ingestion; fall back to no-scale-filter.
                pass

        if position is not None and rotation is not None and keep.any():
            pos_t = torch.as_tensor(np.asarray(position, dtype=np.float32),
                                    device=means.device, dtype=means.dtype)
            rot_t = torch.as_tensor(np.asarray(rotation, dtype=np.float32),
                                    device=means.device, dtype=means.dtype)
            # World -> camera (camera looks +Z, consistent with OccupancyMap.integrate yaw).
            cam_pts = (means - pos_t) @ rot_t   # equivalent to R^T @ (p - t)
            z_cam = cam_pts[:, 2]
            in_range = (z_cam >= self.obs_depth_min) & (z_cam <= self.occupancy_max_range_m)
            # Frustum check using normalized intrinsics (fx,fy in [0,1] image units, cx=cy=0.5).
            fx, fy = float(self.camera.fx), float(self.camera.fy)
            cx, cy = float(self.camera.cx), float(self.camera.cy)
            safe_z = torch.where(z_cam.abs() > 1e-6, z_cam, torch.ones_like(z_cam))
            u = fx * cam_pts[:, 0] / safe_z + cx
            v = fy * cam_pts[:, 1] / safe_z + cy
            # Occupancy-only border shrink: when obs_filter_border_gaussians is True, exclude
            # gaussians originating from the outermost `occupancy_border_px` pixels, where
            # depth predictions are least reliable and produce radial streaks.
            if self.obs_filter_border_gaussians and self.occupancy_border_px > 0:
                w_img, h_img = int(self.camera.w), int(self.camera.h)
                bu = float(self.occupancy_border_px) / max(w_img, 1)
                bv = float(self.occupancy_border_px) / max(h_img, 1)
            else:
                bu = bv = 0.0
            in_fov = (u >= bu) & (u <= 1.0 - bu) & (v >= bv) & (v <= 1.0 - bv)
            keep = keep & in_range & in_fov

        pcd = means[keep].cpu().numpy().astype(np.float32)
        # pcd[:, 1] = -pcd[:, 1]

        # Occlusion-aware filter: drop diffusion-imagined gaussians that lie BEHIND an
        # already-known obstacle (cell-wise raycast on the existing X-Z occupancy grid).
        # Without this, the diffusion model's hallucinated through-wall content gets
        # stamped as obstacle in regions the agent can never raycast-clear.
        if (
            self.occlusion_filter_enabled
            and position is not None
            and pcd.shape[0] > 0
            and self.obs_occupancy is not None
            and self.obs_occupancy.occupancy is not None
        ):
            occ_grid = self.obs_occupancy.occupancy  # (nz, nx); 1=obstacle
            res = float(self.obs_occupancy.resolution)
            x_min = float(self.obs_occupancy.x_min)
            z_min = float(self.obs_occupancy.z_min)
            nx = int(self.obs_occupancy.nx)
            nz = int(self.obs_occupancy.nz)
            ox, _, oz = float(position[0]), float(position[1]), float(position[2])
            sc = int(np.clip(np.floor((ox - x_min) / res), 0, nx - 1))
            sr = int(np.clip(np.floor((oz - z_min) / res), 0, nz - 1))
            keep_mask = np.ones(pcd.shape[0], dtype=bool)
            buf = int(getattr(self, "occlusion_target_buffer_cells", 3))
            for i in range(pcd.shape[0]):
                tc = int(np.clip(np.floor((pcd[i, 0] - x_min) / res), 0, nx - 1))
                tr = int(np.clip(np.floor((pcd[i, 2] - z_min) / res), 0, nz - 1))
                # DDA / Bresenham over (col, row); check intermediate cells (exclude endpoints).
                dr = tr - sr
                dc = tc - sc
                steps = max(abs(dr), abs(dc))
                if steps <= 1 + buf:
                    continue
                blocked = False
                # stop `buf` cells short of the target so we don't reject points
                # belonging to the same wall surface as already-stamped obstacle
                # cells (curtain self-rejection).
                for s in range(1, steps - buf):
                    rr = sr + int(round(dr * s / steps))
                    cc = sc + int(round(dc * s / steps))
                    if 0 <= rr < nz and 0 <= cc < nx and occ_grid[rr, cc] == 1:
                        blocked = True
                        break
                if blocked:
                    keep_mask[i] = False
            pcd = pcd[keep_mask]
        # if self.inc_pcd_flip_y:
        #     pcd[:, 1] = -pcd[:, 1]
        return pcd  # (N, 3)

    def _extract_obs_colored_pcd(self):
        gaussians = self.obs_scene.float()
        obs_pcd = self._gaussians_to_o3d_pcd(
            gaussians,
        )
        return obs_pcd

    def _extract_augmented_colored_pcd(self):
        gaussians = self.augmented_scene.float()
        augmented_pcd = self._gaussians_to_o3d_pcd(
            gaussians,
        )
        return augmented_pcd

    def _extract_colored_pcd(self, scene: Gaussians):
        gaussians = scene.float()
        colored_pcd = self._gaussians_to_o3d_pcd(
            gaussians,
        )
        return colored_pcd

    def _gaussians_to_o3d_pcd(
        self,
        gaussians,
        *,
        b: int = 0,
        opacity_thresh: float = 1e-3,
        premultiplied_Y00: bool = False,
        color_mapping: str = "sigmoid",   # "sigmoid" | "clip" | "none"
    ) -> o3d.geometry.PointCloud:
        # sanity checks
        assert hasattr(gaussians, "means") and hasattr(gaussians, "harmonics") and hasattr(gaussians, "opacities")
        means = gaussians.means
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities

        assert means.dim() == 3 and means.size(-1) == 3, "means should be (B, N, 3)"
        assert harmonics.dim() == 4 and harmonics.size(2) == 3, "harmonics should be (B, N, 3, d_sh)"
        assert opacities.dim() == 2, "opacities should be (B, N)"
        assert 0 <= b < means.size(0), f"batch index {b} out of range"

        # points
        pts = means[b]  # (N, 3)

        # base RGB from SH DC (index 0 along last dim)
        base_rgb = harmonics[b, :, :, 0]  # (N, 3)
        if not premultiplied_Y00:
            Y00 = 0.28209479177387814  # sqrt(1/(4π))
            base_rgb = base_rgb * Y00

        # color mapping
        if color_mapping == "sigmoid":
            base_rgb = torch.sigmoid(base_rgb)
        elif color_mapping == "clip":
            base_rgb = base_rgb.clamp(0.0, 1.0)
        elif color_mapping == "none":
            pass
        else:
            raise ValueError("color_mapping must be one of {'sigmoid','clip','none'}")

        # opacity filtering
        keep = opacities[b] > opacity_thresh
        if keep.numel() != pts.shape[0]:
            raise ValueError("Mismatch between opacities and means along N")
        pts = pts[keep]
        cols = base_rgb[keep]

        # to Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.detach().cpu().to(torch.float64).numpy())
        pcd.colors = o3d.utility.Vector3dVector(cols.detach().cpu().to(torch.float64).numpy())
        return pcd

    def render_image(
        self,
        extrinsics: Optional[Tensor] = None,              # (4, 4)
        intrinsics: Optional[Tensor] = None,
        near: Optional[Union[float, Tensor]] = None,
        far: Optional[Union[float, Tensor]] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        filter_ceiling: Optional[bool] = False, 
        ceiling_threshold: Optional[float] = 2.5,
    ):
        # Fallback to self attributes if arguments are not provided
        if intrinsics is None:
            intrinsics = torch.tensor(self.camera.intrinsics).to("cuda")
        if near is None:
            near = torch.tensor(self.camera.near).to("cuda")
        if far is None:
            far = torch.tensor(self.camera.far).to("cuda")
        if h is None or w is None:
            h, w = self.camera.h, self.camera.w

        output = self.model.ema.ema_model.model.render(self.augmented_scene, 
                                              extrinsics.to("cuda"), 
                                              intrinsics, near, far, h, w,
                                              filter_ceiling, ceiling_threshold)
        return output
    
    def render_video(
        self,
        render_poses: List[Tensor],
        intrinsics: Optional[Tensor] = None,
        near: Optional[Union[float, Tensor]] = None,
        far: Optional[Union[float, Tensor]] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ):
        # Fallback to self attributes if arguments are not provided
        if intrinsics is None:
            intrinsics = torch.tensor(self.camera.intrinsics)
        if near is None:
            near = torch.tensor(self.camera.near)
        if far is None:
            far = torch.tensor(self.camera.far)
        if h is None or w is None:
            h, w = self.camera.h, self.camera.w

        rgb_frames = []
        depth_frames = []
        semantics = []
        for _, pose in enumerate(render_poses):
            rgb, depth, semantic = self.render_image(pose, intrinsics, near, far, h, w)
            rgb_frames.append(rgb)
            depth_frames.append(depth)
            if semantic is not None:
                semantics.append(semantic)
        return rgb_frames, depth_frames, semantics

    def render_any_scene_image(
        self,
        augmented_scene: Gaussians,
        extrinsics: Optional[Tensor] = None,              # (4, 4)
        intrinsics: Optional[Tensor] = None,
        near: Optional[Union[float, Tensor]] = None,
        far: Optional[Union[float, Tensor]] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        filter_ceiling: Optional[bool] = False, 
        ceiling_threshold: Optional[float] = 2.5,
        query_label: Optional[str] = None,  # for semantic rendering
    ):
        # Fallback to self attributes if arguments are not provided
        if intrinsics is None:
            intrinsics = torch.tensor(self.camera.intrinsics).to("cuda")
        if near is None:
            near = torch.tensor(self.camera.near).to("cuda")
        if far is None:
            far = torch.tensor(self.camera.far).to("cuda")
        if h is None or w is None:
            h, w = self.camera.h, self.camera.w

        output = self.model.ema.ema_model.model.render(augmented_scene, 
                                              extrinsics.to("cuda"), intrinsics, near, far, h, w,
                                              filter_ceiling, ceiling_threshold,
                                              query_label=query_label)
        return output


    def render_any_scene_video(
        self,
        augmented_scene: Gaussians,
        render_poses: List[Tensor],
        intrinsics: Optional[Tensor] = None,
        near: Optional[Union[float, Tensor]] = None,
        far: Optional[Union[float, Tensor]] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
        query_label: Optional[str] = None,  # for semantic rendering
    ):
        # Fallback to self attributes if arguments are not provided
        if intrinsics is None:
            intrinsics = torch.tensor(self.camera.intrinsics)
        if near is None:
            near = torch.tensor(self.camera.near)
        if far is None:
            far = torch.tensor(self.camera.far)
        if h is None or w is None:
            h, w = self.camera.h, self.camera.w

        rgb_frames = []
        depth_frames = []
        semantics = []
        for _, pose in enumerate(render_poses):
            rgb, depth, semantic = self.render_any_scene_image(augmented_scene, pose, intrinsics, near, far, h, w, query_label=query_label)
            rgb_frames.append(rgb)
            depth_frames.append(depth)
            if semantic is not None:
                semantics.append(semantic)
        return rgb_frames, depth_frames, semantics
    
    def export_scene(
        self,
        path: Path,
        extrinsics: Tensor,
        *,
        opacity_thresh: float = 0.1,
        max_scale_m: float = 0.15,
        knn_k: int = 8,
        knn_radius_m: float = 0.25,
    ):
        """Export the augmented gaussian scene to a 3DGS-compatible PLY.

        Filters out floating / spurious gaussians before writing:
          - low opacity   (< ``opacity_thresh``)
          - oversized     (max eigenvalue of covariance > ``max_scale_m``^2)
          - isolated      (k-th nearest neighbour distance > ``knn_radius_m``)

        Set ``knn_k <= 0`` or ``knn_radius_m <= 0`` to skip the isolation filter.
        """
        gaussians = self.augmented_scene.float()
        means = gaussians.means          # (B, N, 3)
        covs = gaussians.covariances     # (B, N, 3, 3)
        harmonics = gaussians.harmonics  # (B, N, 3, d_sh)
        opacities = gaussians.opacities  # (B, N)

        if means.numel() == 0 or means.shape[1] == 0:
            return

        keep = opacities >= opacity_thresh  # (B, N)
        # Drop any NaN/Inf gaussians up front so downstream linalg is well-defined.
        finite_mask = (
            torch.isfinite(means).all(dim=-1)
            & torch.isfinite(covs).reshape(*covs.shape[:2], -1).all(dim=-1)
            & torch.isfinite(opacities)
        )
        keep = keep & finite_mask

        # Filter oversized gaussians (use largest eigenvalue of covariance).
        if max_scale_m is not None and max_scale_m > 0 and keep.any():
            # Symmetrize and run on CPU to avoid flaky cusolver batched eigvalsh.
            covs_sym = 0.5 * (covs + covs.transpose(-1, -2))
            try:
                evals = torch.linalg.eigvalsh(covs_sym)
            except Exception:
                evals = torch.linalg.eigvalsh(covs_sym.detach().cpu()).to(covs.device)
            max_var = evals.clamp_min(0).max(dim=-1).values  # (B, N)
            keep = keep & (max_var <= float(max_scale_m) ** 2)

        # Per-batch isolation filter.
        if knn_k and knn_k > 0 and knn_radius_m and knn_radius_m > 0:
            B = means.shape[0]
            iso_mask = torch.zeros_like(keep)
            for b in range(B):
                m = means[b][keep[b]]  # (M, 3)
                if m.shape[0] <= knn_k:
                    iso_mask[b][keep[b]] = True
                    continue
                # pairwise distances; use chunks to avoid huge memory
                d = torch.cdist(m, m)  # (M, M)
                topk = d.topk(knn_k + 1, largest=False).values[:, 1:]  # exclude self
                kth = topk[:, -1]
                local_keep = kth <= float(knn_radius_m)
                idx = torch.nonzero(keep[b], as_tuple=False).squeeze(-1)
                iso_mask[b, idx[local_keep]] = True
            keep = iso_mask

        if not keep.any():
            return

        # Filter to a per-batch list (export_gaussians_to_ply expects (B, N, ...) with
        # equal N across the batch; for B=1 we just slice).
        B = means.shape[0]
        assert B == 1, "export_scene currently assumes batch size 1"
        m = keep[0]
        filtered = Gaussians(
            means=means[:, m].contiguous(),
            covariances=covs[:, m].contiguous(),
            harmonics=harmonics[:, m].contiguous(),
            opacities=opacities[:, m].contiguous(),
            features=None,
        )

        export_gaussians_to_ply(
            filtered,
            extrinsics.to("cuda"),
            path,
        )
    
