from __future__ import annotations

import importlib
import sys
import os
from pathlib import Path
from PIL import Image
from typing import Union, List
from typing import Any, Sequence, Optional

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

def instantiate_dfot_model_wrapper(
    dfot_repo: Union[str, Path],
    overrides: Optional[List[str]] = None,
    ckpt_path: Optional[str] = None,
    config_dirname: str = "configurations",
    config_name: str = "inference",
) -> object:
    """
    Instantiate ModelWrapper from the DFoT project (another repo) by composing its Hydra config.

    Args:
        dfot_repo: Path to the DFoT repo root (the one containing `configurations/` and `model_wrapper.py`).
        overrides: Hydra overrides list, e.g. ["dataset=habitat", "algorithm=dfot_video_pose", "experiment=video_generation"].
        ckpt_path: Optional checkpoint path to override cfg.ckpt_path.
        config_dirname: Name of the config directory inside dfot_repo.
        config_name: Root config file name (without .yaml), e.g. "config".

    Returns:
        model_wrapper: Instantiated ModelWrapper object from the DFoT repo.
    """
    dfot_repo = Path(dfot_repo).resolve()
    config_dir = dfot_repo / config_dirname
    if not config_dir.exists():
        raise FileNotFoundError(f"Hydra config dir not found: {config_dir}")

    # Make sure we can import DFoT modules (model_wrapper, datasets, dfot_utils, etc.)
    if str(dfot_repo) not in sys.path:
        sys.path.insert(0, str(dfot_repo))

    # If the caller already initialized Hydra elsewhere, reset it to avoid:
    # "Hydra is already initialized" errors.
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    overrides = overrides or []

    # Compose DFoT config WITHOUT changing your working directory
    with hydra.initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg: DictConfig = hydra.compose(config_name=config_name, overrides=overrides)

    # Override ckpt if requested
    if ckpt_path is not None:
        cfg.ckpt_path = ckpt_path

    # # Resolve interpolations (optional but often helpful)
    # OmegaConf.resolve(cfg)

    # Import after sys.path is set
    from model_wrapper import ModelWrapper  # from the DFoT repo

    # Instantiate wrapper (DFoT code expects (cfg, ckpt_path))
    wrapper = ModelWrapper(cfg, cfg.ckpt_path)
    return wrapper

class DFoTVGGTModel(BaseWorldModel):
    """Wrapper around DFoT's ModelWrapper for VGGT world model."""

    def __init__(
        self,
        obs_occupancy: OccupancyMap,
        belief_occupancy: OccupancyMap,
        camera: Camera,
        adjacent_angle: float,
        adjacent_distance: float,
        dfot_model_checkpoint: str,
        dfot_repo_path: str,
        dfot_model_image_size: int = 256,
        vggt_model_checkpoint: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.obs_occupancy = obs_occupancy
        self.belief_occupancy = belief_occupancy
        self.dfot_model_checkpoint = dfot_model_checkpoint
        self.dfot_repo_path = dfot_repo_path
        self.vggt_model_checkpoint = vggt_model_checkpoint
        self.dfot_model_image_size = dfot_model_image_size
        self.camera = camera
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
        self._build_vggt_model()
        self._build_dfot_model()
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
    
    def _build_dfot_model(self) -> Any:
        self.dfot_model = instantiate_dfot_model_wrapper(
            dfot_repo=self.dfot_repo_path,
            overrides=[
                "+name=dfot_spoc",
                "dataset=spoc",
                "algorithm=dfot_video_pose",
                "experiment=video_generation",
                "++algorithm={diffusion:{is_continuous:True,precond_scale:0.125},backbone:{use_fourier_noise_embedding:True}}",
            ],
            ckpt_path=self.dfot_model_checkpoint,
        )
        self.dfot_model.algo.to(self.device)
        self.dfot_model.algo.eval()
        print("DFoT model loaded on", self.device)
    
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
            rgb_tensor = (
                torch.tensor(
                    rgb.astype(np.float32)
                ).permute(2, 0, 1)
                / 255.0
            )
            rgb_tensor = F.interpolate(
                rgb_tensor.unsqueeze(0),
                size=(self.dfot_model_image_size, self.dfot_model_image_size),
                mode="bilinear",
                antialias=True,
            )[0]
            self.current_rgb = self.dfot_model.algo._normalize_x(rgb_tensor.to(self.device))
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

        intrinsics_flat = [self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy]  # fx, fy, cx, cy
        # take inverse to get world to camera
        key_frame_poses = [torch.linalg.inv(pose) for pose in key_frame_poses]
        # reshape each pose to (3, 4) and flatten
        key_frame_poses_flat = [pose[:3, :].reshape(-1) for pose in key_frame_poses]  # (num_key_frames, 12)
        # convert to torch tensor
        key_frame_poses_tensor = torch.stack(key_frame_poses_flat, dim=0).to(self.device)  # (T, 12)
        # [fx,fy,cx,cy] + 12*E -> (T,16)
        conds = torch.cat(
            [
                torch.tensor(intrinsics_flat, device=self.device).unsqueeze(0).repeat(len(key_frame_poses_tensor), 1),
                key_frame_poses_tensor,
            ],
            dim=1,
        ).unsqueeze(0)  # (1, T, 16)
        # videos in inp is the self.current_rgb repeated num_key_frames times
        videos = self.current_rgb.unsqueeze(0).unsqueeze(0).repeat(1, num_key_frames, 1, 1, 1)  # (1, T, 3, H, W)
        inp = {
            "videos": videos,
            "conds": conds,
        }
        output = self.dfot_model.inference(inp)
        return output

    @with_timing
    def _imagine_along_path(
        self, 
        imagine_poses: List[Union[Tensor, np.ndarray]], 
        return_full_video: bool=False, 
        return_colored_pcd: bool=False, 
        num_key_frames: int=8,
        query_label: Optional[str]=None,
    ) -> Any:
        """Imagine in place given a goal pose, return key frame renders and optionally full video and belief scenes."""
        imagine_poses = [np.linalg.inv(self.initial_location['pose']) @ pose for pose in imagine_poses]
        imagine_poses = torch.stack([Tensor(pose) for pose in imagine_poses], dim=0).to(self.device)
        
        render_poses = imagine_poses
        
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
        extra_frames = num_key_frames - len(key_frame_poses)
        while len(key_frame_poses) < num_key_frames:
            key_frame_poses.append(key_frame_poses[-1])
        intrinsics_flat = [self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy]  # fx, fy, cx, cy
        # take inverse to get world to camera
        key_frame_poses = [torch.linalg.inv(pose) for pose in key_frame_poses]
        # reshape each pose to (3, 4) and flatten
        key_frame_poses_flat = [pose[:3, :].reshape(-1) for pose in key_frame_poses]  # (num_key_frames, 12)
        # convert to torch tensor
        key_frame_poses_tensor = torch.stack(key_frame_poses_flat, dim=0).to(self.device)  # (T, 12)
        # [fx,fy,cx,cy] + 12*E -> (T,16)
        conds = torch.cat(
            [
                torch.tensor(intrinsics_flat, device=self.device).unsqueeze(0).repeat(len(key_frame_poses_tensor), 1),
                key_frame_poses_tensor,
            ],
            dim=1,
        ).unsqueeze(0)  # (1, T, 16)
        # videos in inp is the self.current_rgb repeated num_key_frames times
        videos = self.current_rgb.unsqueeze(0).unsqueeze(0).repeat(1, num_key_frames, 1, 1, 1)  # (1, T, 3, H, W)
        inp = {
            "videos": videos,
            "conds": conds,
        }
        key_video = self.dfot_model.inference(inp)

        key_video = key_video[0]  # (T, 3, H, W)
        # remove extra frames if any
        if extra_frames > 0:
            key_video = key_video[:-extra_frames]
        key_rgbs = [key_video[i].permute(1, 2, 0).cpu().numpy() for i in range(len(key_video))]
        # convert to unit8
        key_rgbs_255 = [(rgb * 255).astype(np.uint8) for rgb in key_rgbs]
        # Run VGGT inference to get the depth maps for the key frames
        with torch.no_grad():
            with torch.amp.autocast(dtype=self.vggt_model_dtype):
                images = self._preprocess_images(key_rgbs_255)
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

            # scale depths
            depth_map_scaled = depth_map / s_depth
            key_depths = deepcopy(depth_map_scaled)[0].permute(0, 3, 1, 2)    # (T,1,H,W)
            # resize depth maps to original image size
            key_depths = F.interpolate(
                key_depths,
                size=(key_rgbs[0].shape[0], key_rgbs[0].shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            # to list of numpy
            key_depths = [key_depths[i].squeeze(-1).cpu().numpy() for i in range(len(key_depths))]
            if return_colored_pcd:
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
                colored_pcd = pcd
            else:
                colored_pcd = None
        
        # convert key_rgbs to (H,W,3) uint8
        key_rgbs = [(rgb * 255).astype(np.uint8) for rgb in key_rgbs]
        key_rgbs = [np.clip(rgb, 0, 255) for rgb in key_rgbs]
        key_output = RenderOutput(rgb=key_rgbs, depth=key_depths, pose=key_frame_poses)
        full_output = deepcopy(key_output)

        return key_output, full_output, colored_pcd

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
            images.append(video[0][-4].permute(1, 2, 0).cpu().numpy())  # (H, W, 3)
            self._metrics["model_inference_time_imagine"] += exe_time
        render_output = RenderOutput(rgb=images)

        return render_output