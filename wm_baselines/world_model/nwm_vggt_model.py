from __future__ import annotations

import importlib
import sys
import os
from pathlib import Path
from PIL import Image
from typing import Union, List
from typing import Any, Sequence, Optional
from torchvision import transforms

import yaml
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
import open3d as o3d
from copy import deepcopy

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from belief_baselines.utils.data_classes import Frame, RenderOutput
from belief_baselines.agent.perception.camera import Camera
from belief_baselines.agent.perception.occupancy import OccupancyMap
from belief_baselines.world_model.base_world_model import BaseWorldModel
from belief_baselines.utils.planning_utils import rotation_angle
from belief_baselines.utils.common_utils import with_timing
from belief_baselines.utils.vision_utils import (
    interpolate_pose_wobble,
    get_yaw_from_pose,
    to_local_coords,
    normalize_data,
    angle_difference,
    get_delta_np,
    _ensure_time_dim,
    _slice_per_frame,
    _to_hw3,
    _as_4x4,
    _depth_valid_mask_per_frame,
    _conf_flat_per_frame,
    _mask_take,
    _get_T_c1_for_frame,
    _transform_points_np,
    _image_to_hw3,
    _stack_points_colors,
    _move_time_from_dim1_to_dim0,
    _align_time_dim_like,
    _as_tensor3x3,
    _depth_scale_from_intrinsics
)

ACTION_STATS = {
    "min": [-2.5, -4], # [min_dx, min_dy]
    "max": [5, 4] # [max_dx, max_dy]
}
for key in ACTION_STATS:
    ACTION_STATS[key] = np.expand_dims(ACTION_STATS[key], axis=0)

def instantiate_nwm_wrapper(
    nwm_repo: Union[str, Path],
    ckpt_path: Optional[str] = None,
    config_dirname: str = "config",
    config_name: str = "eval_config.yaml",
) -> object:
    """Instantiate NWM model from repo and checkpoint.

    Args:
        nwm_repo (Union[str, Path]): Path to the NWM repository.
        config_dirname (str, optional): Name of the config directory. Defaults to "config".
        config_name (str, optional): Name of the config file. Defaults to "eval_config.yaml".

    Returns:
        object: Instantiated NWM model.
    """
    nwm_repo = str(nwm_repo)
    sys.path.insert(0, nwm_repo)

    config_dir = os.path.join(nwm_repo, config_dirname)
    config_path = os.path.join(config_dir, config_name)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if ckpt_path is not None:
        config['ckpt_path'] = ckpt_path
    
    from isolated_nwm_infer import ModelWrapper
    nwm_model = ModelWrapper(
        config=config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    return nwm_model

def compute_actions(traj_data, len_traj_pred=8, normalize=True, metric_waypoint_spacing=0.2):
    start_index = 0
    yaw = traj_data["yaw"][start_index:]
    positions = traj_data["position"][start_index:]

    if len(yaw.shape) == 2:
        yaw = yaw.squeeze(1)

    if yaw.shape != (len_traj_pred,):
        raise ValueError("is used?")

    waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
    waypoints_yaw = angle_difference(yaw[0], yaw)
    actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
    actions = actions[1:]
    
    if normalize:
        actions[:, :2] /= metric_waypoint_spacing
    
    return actions    

class NWMVGGTModel(BaseWorldModel):
    """Navigation World Model with VGGT backbone for 3D perception.

    This class integrates a Navigation World Model (NWM) with a VGGT backbone to process
    visual observations and update the world model's belief state.

    Args:
        nwm_model (object): An instantiated NWM model.
        vggt_model (VGGT): An instantiated VGGT model.
        camera (Camera): Camera parameters for rendering.
        occupancy_map (OccupancyMap): Occupancy map for the environment.
    """

    def __init__(
        self,
        obs_occupancy: OccupancyMap,
        belief_occupancy: OccupancyMap,
        camera: Camera,
        adjacent_angle: float,
        adjacent_distance: float,
        nwm_checkpoint: str,
        nwm_repo_path: Union[str, Path],
        nwm_image_size: int = 224,
        vggt_model_checkpoint: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.nwm_image_size = nwm_image_size
        self.nwm_checkpoint = nwm_checkpoint
        self.nwm_repo_path = nwm_repo_path
        self.vggt_model_checkpoint = vggt_model_checkpoint
        self.camera = camera
        self.obs_occupancy = obs_occupancy
        self.belief_occupancy = belief_occupancy
        self.nwm_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        self.nwm_unnormalize = transforms.Normalize(
            mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
            std=[1 / 0.5, 1 / 0.5, 1 / 0.5]
        )

        self.initial_location = {}
        self.step = -1
        self._metrics = {
            "model_inference_time": 0.0,
            "update_occupancy_time": 0.0,
            "model_inference_time_imagine": 0.0,
            "obs_steps": -1,
        }
        self.rgb_images: List[np.ndarray] = []
        self.scene_pcd: o3d.geometry.PointCloud = None
        self.resample_pcd = True
        self.resample_pcd_check_dict = {"z_previous": None, "t_previous": None}
        self.adjacent_angle = adjacent_angle
        self.adjacent_distance = adjacent_distance

        self._build_nwm()
        self._build_vggt_model()

        self.current_rgb: Tensor = None
        self.current_pose: Tensor = None

    def reset(self):
        resolution = self.obs_occupancy.resolution
        obstacle_height_thresh = self.obs_occupancy.obstacle_height_thresh
        self.obs_occupancy = OccupancyMap(resolution, obstacle_height_thresh)
        resolution = self.belief_occupancy.resolution
        obstacle_height_thresh = self.belief_occupancy.obstacle_height_thresh
        self.belief_occupancy = OccupancyMap(resolution, obstacle_height_thresh)
        self.initial_location = {}
        self.step = -1
        self._metrics = {
            "model_inference_time": 0.0,
            "update_occupancy_time": 0.0,
            "model_inference_time_imagine": 0.0,
            "obs_steps": -1,
        }
        self.rgb_images = []
        self.scene_pcd = None
        self.resample_pcd = True
        self.resample_pcd_check_dict = {"z_previous": None, "t_previous": None}
        self.current_rgb = None
        self.current_pose = None

    def _build_vggt_model(self) -> Any:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vggt_model_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if self.vggt_model_checkpoint is None:
            self.vggt_model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        else:
            self.vggt_model = VGGT().to(self.device).eval()
            ckpt = torch.load(self.vggt_model_checkpoint, map_location=self.device)
            if "state_dict" in ckpt:
                self.vggt_model.load_state_dict(ckpt["state_dict"], strict=False)
            elif "model" in ckpt:
                self.vggt_model.load_state_dict(ckpt["model"], strict=False)
            else:
                self.vggt_model.load_state_dict(ckpt, strict=False)
        print("VGGT model loaded on", self.device)
    
    def _build_nwm(self) -> Any:
        self.nwm_model = instantiate_nwm_wrapper(
            nwm_repo=self.nwm_repo_path,
            ckpt_path=self.nwm_checkpoint
        )
    
    def _poses_to_traj(self, poses) -> Dict[str, Any]:
        yaws, positions = [], []
        for pose in poses:
            yaw = get_yaw_from_pose(pose)
            position = np.array([pose[2, 3], pose[0, 3]])
            yaws.append(yaw)
            positions.append(position)
        traj = {
            "yaw": np.array(yaws, dtype=np.float64), 
            "position": np.array(positions, dtype=np.float32)
        }
        return traj

    def _compute_poses(self, model_input, n: int) -> torch.Tensor:
        near = model_input["near"]
        far = model_input["far"]

        render_poses = torch.stack(
            [
                interpolate_pose_wobble(
                    model_input["ctxt_c2w"][0][0],
                    model_input["trgt_c2w"][0][0],
                    t / n,
                    wobble=False,
                )
                for t in range(n)
            ],
            0,
        )

        return render_poses

    def update_observation(self, observation: Dict[str, Any], force_update: bool) -> Dict[str, Any]:
        """Update the occupancy map with a new observation."""
        self.step += 1
        self._metrics["obs_steps"] += 1
        rgb = observation.get("rgb")
        pose = observation.get("pose")  # (4, 4) world to camera

        if self.initial_location.get('pose') is None:
            self.initial_location['pose'] = pose
        
        if self.initial_location.get('pose') is not None:
            pose_map = np.linalg.inv(self.initial_location['pose']) @ pose # make the initial pose the origin
            position = pose_map[:3, 3]
            rotation = pose_map[:3, :3]
        
        self._update_resample_pcd(pose, force=force_update)
        if self.resample_pcd:
            self.rgb_images.append(rgb)
            self.resample_pcd = False
            # Prepare obs
            rgb_pil = Image.fromarray(rgb)
            rgb_tensor = self.nwm_transform(rgb_pil)
            self.current_rgb = rgb_tensor.to(self.device)
            # Prepare pose
            pose_map_tensor = torch.tensor(pose_map, dtype=torch.float32)
            self.current_pose = pose_map_tensor.to(self.device)

            # Run VGGT inference to get scene point cloud
            self.scene_pcd, exe_time = self._inference_pcd()

            self._metrics["model_inference_time"] += exe_time
            previous_map = deepcopy(self.obs_occupancy) if self.step > 0 else None
            resolution = self.obs_occupancy.resolution
            obstacle_height_thresh = self.obs_occupancy.obstacle_height_thresh
            self.obs_occupancy = OccupancyMap(resolution, obstacle_height_thresh)
            _, exe_time = self.obs_occupancy.integrate(
                np.array(self.scene_pcd.points), 
                position, 
                rotation, 
                intrinsics=self.camera.intrinsics,
                prev_free_map=previous_map
            )
            self._metrics["update_occupancy_time"] += exe_time
        
        assets = {
            "rgb": rgb,
            "occupancy": self.obs_occupancy,
            "position": position,
            "rotation": rotation,
            "pose": pose_map,
            "initial_pose": self.initial_location['pose'],
        }
        return assets

    def _preprocess_images(self, image_list: List[np.ndarray]) -> Tensor:
        # save images to temp folder and load using VGGT utils
        temp_dir = f"/tmp/vggt_temp_images_{os.getpid()}"
        os.makedirs(temp_dir, exist_ok=True)
        image_paths = []
        for idx, img in enumerate(image_list):
            img_path = os.path.join(temp_dir, f"img_{idx:03d}.png")
            Image.fromarray(img).save(img_path)
            image_paths.append(img_path)
        images_tensor = load_and_preprocess_images(image_paths).to(self.device)
        return images_tensor
    
    def _make_pcd_from_pointmap(
        self,
        point_map,                   # (T,H,W,3) np or torch OR (H,W,3)
        image=None,                  # (T,3,H,W) torch OR (3,H,W)
        conf_map=None,               # (T,H,W) / (1,T,H,W) / (T,H,W,1) / (H,W)
        depth_map=None,              # (T,H,W) / (T,H,W,1) / (H,W)
        ref_extrinsic_first_cw: torch.Tensor = None,  # (3,4) or (4,4)
        current_extrinsic_cw: torch.Tensor | None = None,  # (T,3,4) or (T,4,4) if needed
        points_are_world_frame: bool = True,
        voxel_downsample: float = 0.0,
        conf_thresh: float = 0.3,
    ) -> o3d.geometry.PointCloud:
        """
        Builds a single Open3D PCD by stacking T frames and expressing all points in the FIRST frame's camera coords.
        """
        if ref_extrinsic_first_cw is None:
            raise ValueError("ref_extrinsic_first_cw is required.")

        pmT, T_pm = _ensure_time_dim(point_map, "point_map")
        imgT, T_img = _ensure_time_dim(image, "image") if image is not None else (None, 0)
        cmT,  T_cm  = _ensure_time_dim(conf_map, "conf_map") if conf_map is not None else (None, 0)
        dmT,  T_dm  = _ensure_time_dim(depth_map, "depth_map") if depth_map is not None else (None, 0)

        imgT = _align_time_dim_like(pmT, imgT)
        cmT  = _align_time_dim_like(pmT, cmT)
        dmT  = _align_time_dim_like(pmT, dmT)

        T = T_pm
        T_img = 0 if imgT is None else imgT.shape[0]
        T_cm  = 0 if cmT  is None else cmT.shape[0]
        T_dm  = 0 if dmT  is None else dmT.shape[0]

        if T_img and T_img not in (0, T): raise ValueError(f"image T={T_img} mismatch with point_map T={T}")
        if T_cm  and T_cm  not in (0, T): raise ValueError(f"conf_map T={T_cm} mismatch with point_map T={T}")
        if T_dm  and T_dm  not in (0, T): raise ValueError(f"depth_map T={T_dm} mismatch with point_map T={T}")

        if not points_are_world_frame:
            if current_extrinsic_cw is None:
                raise ValueError("current_extrinsic_cw required when points_are_world_frame=False.")
            if current_extrinsic_cw.ndim == 4 and current_extrinsic_cw.shape[0] == 1:
                current_extrinsic_cw = current_extrinsic_cw.squeeze(0)  # (T,3,4)/(T,4,4)

        all_pts, all_cols = [], []

        for t in range(T):
            pm_t = _slice_per_frame(pmT, t)               # (H,W,3) or (3,H,W) torch/np
            if isinstance(pm_t, np.ndarray):
                pm_t = torch.from_numpy(pm_t)
            pts_flat, (H, W) = _to_hw3(pm_t)

            # Build masks
            mask = None
            if cmT is not None:
                cm_t = _slice_per_frame(cmT, t)
                cm_1d = _conf_flat_per_frame(cm_t, H, W)
                mask = (cm_1d >= conf_thresh)

            if dmT is not None:
                dm_t = _slice_per_frame(dmT, t)
                good_d = _depth_valid_mask_per_frame(dm_t, H, W)
                mask = good_d if mask is None else (mask & good_d)

            pts_flat = _mask_take(pts_flat, mask)

            # Transform to first-camera coords
            E_curr = None
            if not points_are_world_frame:
                E_curr = current_extrinsic_cw[t]
            T_c1_for_t = _get_T_c1_for_frame(ref_extrinsic_first_cw, points_are_world_frame, E_curr)
            pts_flat = _transform_points_np(pts_flat, T_c1_for_t)

            # Colors
            cols_flat = None
            if imgT is not None:
                img_t = _slice_per_frame(imgT, t)   # (3,H,W) or (H,W,3)
                cols_all = _image_to_hw3(img_t)
                cols_flat = _mask_take(cols_all, mask)

            all_pts.append(pts_flat.astype(np.float32, copy=False))
            all_cols.append(cols_flat.astype(np.float32, copy=False) if cols_flat is not None else None)

        # Stack all frames
        pts, cols = _stack_points_colors(all_pts, all_cols)

        # Build Open3D cloud
        pcd = o3d.geometry.PointCloud()
        if pts.size == 0:
            return pcd
        pcd.points = o3d.utility.Vector3dVector(pts)
        if cols is not None and cols.shape[0] == pts.shape[0]:
            pcd.colors = o3d.utility.Vector3dVector(cols)

        if voxel_downsample and voxel_downsample > 0:
            pcd = pcd.voxel_down_sample(voxel_downsample)
        if len(pcd.points) > 0:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
        return pcd

    @with_timing
    def _inference_pcd(self) -> o3d.geometry.PointCloud:
        """Run VGGT model inference to get the scene point cloud."""
        with torch.no_grad():
            with torch.amp.autocast(dtype=self.vggt_model_dtype):
                images = self._preprocess_images(self.rgb_images)
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.vggt_model.aggregator(images)

            # Predict Cameras
            pose_enc = self.vggt_model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])  # extrinsic: (1,T,3,4) or (1,T,4,4); intrinsic: (1,T,3,3) or (1,3,3)

            # SCALE: compute depth scale from intrinsics
            # K_gt must be a 3x3 (OpenCV) intrinsic for the camera
            K_gt = _as_tensor3x3(self.camera.intrinsics, device=intrinsic.device, dtype=intrinsic.dtype)
            K_est_seq = intrinsic.squeeze(0)  # (T,3,3) or (3,3)
            s_depth = _depth_scale_from_intrinsics(K_est_seq, K_gt)

            # Predict Depth Maps
            depth_map, depth_conf = self.vggt_model.depth_head(aggregated_tokens_list, images, ps_idx)

            # scale depths before unprojection
            depth_map_scaled = depth_map / s_depth

            # Construct 3D Points from Depth Maps and Cameras (world coords if the unprojector uses T_cw)
            point_map_by_unprojection = unproject_depth_map_to_point_map(
                depth_map_scaled.squeeze(0),      # (T,H,W,1) -> (T,H,W,1) or (T,H,W)
                extrinsic.squeeze(0),             # (T,3,4)/(T,4,4)
                intrinsic.squeeze(0)              # (T,3,3)   (estimates; OK because we corrected depth)
            )

            img = images.squeeze(0)                # (T,3,H,W)
            pm_world = point_map_by_unprojection   # (T,H,W,3)
            dm = depth_map_scaled.squeeze(0)       # (T,H,W,1)

            # Build PCD in first frame coords
            E_first = _as_4x4(extrinsic[0, 0])     # (4,4) camera-from-world of first frame
            pcd = self._make_pcd_from_pointmap(
                point_map=pm_world,
                image=img,
                conf_map=depth_conf.squeeze(0) if 'depth_conf' in locals() else None,
                depth_map=dm,
                ref_extrinsic_first_cw=E_first,
                points_are_world_frame=True,
                voxel_downsample=0.01,
                conf_thresh=0.6,
            )

        return pcd

    def _update_resample_pcd(self, current_pose: np.ndarray, force: bool = False) -> None:
        """Update whether to resample pcd at each step."""
        if self.step == 0:
            self.resample_pcd_check_dict["z_previous"] = current_pose[:, 2][:3]  # forward vector
            self.resample_pcd_check_dict["t_previous"] = current_pose[:, 3][:3]  # translation
        if force:
            self.resample_pcd = True
            self.resample_pcd_check_dict["z_previous"] = current_pose[:, 2][:3]  # forward vector
            self.resample_pcd_check_dict["t_previous"] = current_pose[:, 3][:3]  # translation
            return
        z_idx = current_pose[:, 2][:3]  # current forward vector
        t_idx = current_pose[:, 3][:3]  # current translation
        angle = rotation_angle(self.resample_pcd_check_dict["z_previous"], z_idx)
        distance = np.linalg.norm(t_idx - self.resample_pcd_check_dict["t_previous"])
        if angle > self.adjacent_angle or distance > self.adjacent_distance:
            self.resample_pcd = True
            self.resample_pcd_check_dict["z_previous"] = z_idx
            self.resample_pcd_check_dict["t_previous"] = t_idx
        
    @with_timing
    def _imagine_in_place(
        self, 
        imagine_goal_pose: Union[Tensor, np.ndarray], 
        num_key_frames: int=8,
    ) -> Any:
        """Imagine in place given a goal pose, return key frame renders and optionally full video and belief scenes."""
        imagine_goal_pose = Tensor(imagine_goal_pose)
        # interpolate imagine poses
        ctxt_c2w = self.current_pose[None, None, :, :].to(self.device)
        trgt_c2w = imagine_goal_pose[None, None, :, :].to(self.device)
        
        inp = {
            "ctxt_c2w": ctxt_c2w,
            "trgt_c2w": trgt_c2w,
            "near": torch.tensor(self.camera.near).to(self.device),
            "far": torch.tensor(self.camera.far).to(self.device),
        }
        render_poses = self._compute_poses(inp, n=50)
        key_frame_indices = [0]
        z_start = render_poses[0][:, 2][:3]
        t_start = render_poses[0][:, 3][:3]
        z_previous = z_start
        t_previous = t_start
        num_frames = len(render_poses)
        for idx in range(1, num_frames):
            current_pose = render_poses[idx]
            z_idx = current_pose[:, 2][:3]  # current forward vector
            t_idx = current_pose[:, 3][:3]  # current translation
            angle = rotation_angle(z_previous.detach().cpu().numpy(), z_idx.detach().cpu().numpy())
            distance = torch.norm(t_idx - t_previous)
            if angle > self.adjacent_angle or distance > self.adjacent_distance or idx==num_frames-1: # must include the last
                key_frame_indices.append(idx)
                z_previous = z_idx
                t_previous = t_idx
        key_frame_poses = [render_poses[idx] for idx in key_frame_indices]
        # if len(key_frame_poses) > num_key_frames, keep the first num_key_frames
        if len(key_frame_poses) > num_key_frames:
            key_frame_poses = key_frame_poses[:num_key_frames]
        # if less, pad by repeating the last one
        while len(key_frame_poses) < num_key_frames:
            key_frame_poses.append(key_frame_poses[-1])
        # convert to numpy
        key_frame_poses = [pose.detach().cpu().numpy() for pose in key_frame_poses]
        traj = self._poses_to_traj(key_frame_poses)
        actions = compute_actions(traj, len_traj_pred=num_key_frames)
        actions[:, :2] = normalize_data(actions[:, :2], ACTION_STATS)
        delta = get_delta_np(actions)
        delta_tensor = torch.as_tensor(delta, dtype=torch.float32).to(self.device)
        # repeat self.current_rgb num_context times for obs so that it becomes (1, num_key_frames, 3, H, W)
        obs = self.current_rgb.unsqueeze(0).unsqueeze(0).repeat(1, num_key_frames//2, 1, 1, 1)
        # rollouts
        video = []
        for i in range(num_key_frames-1):
            curr_delta = delta_tensor[i:i+1]
            x_pred_pixels = self.nwm_model.forward(obs, curr_delta.unsqueeze(0))
            obs = torch.cat([obs, x_pred_pixels.unsqueeze(1)], dim=1)
            obs = obs[:, 1:] # remove first observation
            video.append(x_pred_pixels.squeeze(0))
        # append the observation frame at the beginning
        video = [self.current_rgb] + video  # list of (3,H,W)
        # unnormalize
        video = [self.nwm_unnormalize(frame) for frame in video]
        # detach
        video = [frame.detach().to(torch.float32) for frame in video]
        return video

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
        initial_pose = np.asarray(initial_pose, dtype=np.float64)
        images = []
        semantics = []
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
            video, exe_time = self._imagine_in_place(T_cam2world)
            images.append(video[3].permute(1, 2, 0).cpu().numpy())  # (H, W, 3)
            self._metrics["model_inference_time_imagine"] += exe_time
        render_output = RenderOutput(rgb=images)

        return render_output
