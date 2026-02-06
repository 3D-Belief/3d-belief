from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig
import os
from PIL import Image
import numpy as np
import open3d as o3d
import torch
import imageio
from copy import deepcopy
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from torch import Tensor
from belief_baselines.utils.data_classes import Frame, RenderOutput
from belief_baselines.agent.perception.camera import Camera
from belief_baselines.agent.perception.occupancy import OccupancyMap
from belief_baselines.world_model.base_world_model import BaseWorldModel
from belief_baselines.utils.vision_utils import (
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
from belief_baselines.utils.planning_utils import rotation_angle
from belief_baselines.utils.common_utils import with_timing

class VGGTModel(BaseWorldModel):
    """VGGT world model.

    Responsibilities:
    - hold config reference
    - maintain an occupancy map using VGGT features
    - implement update_observation() to update the occupancy map
    - implement render_image() and render_video() to visualize the occupancy map
    """

    def __init__(
        self, 
        obs_occupancy: OccupancyMap, 
        camera: Camera,
        adjacent_angle: float,
        adjacent_distance: float,
        model_checkpoint: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.obs_occupancy = obs_occupancy
        self.camera = camera
        self.initial_location = {}
        self.step = -1
        self._metrics = {
            "model_inference_time": 0.0,
            "update_occupancy_time": 0.0,
            "obs_steps": -1,
        }
        self.rgb_images: List[np.ndarray] = []
        self.scene_pcd: o3d.geometry.PointCloud = None
        self.resample_pcd = True
        self.resample_pcd_check_dict = {"z_previous": None, "t_previous": None}
        self.adjacent_angle = adjacent_angle
        self.adjacent_distance = adjacent_distance
        self._build_model(model_checkpoint)

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
            # Run VGGT inference to get scene point cloud
            self.scene_pcd, exe_time = self._inference_pcd()
            ## DEBUG
            color, depth = self.render_image(pose_map)
            # save color
            # imageio.imwrite(f"/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/debug/scene_color_{self.step}.png", color)
            # save pcd as .ply
            # o3d.io.write_point_cloud(f"/home/ubuntu/jianwen-us-midwest-1/shulab-jhu/codebase/embodied_tasks/belief_baselines/outputs/debug/scene_{self.step}.ply", self.scene_pcd)
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

    def reset(self):
        """Reset the occupancy maps."""
        resolution = self.obs_occupancy.resolution
        obstacle_height_thresh = self.obs_occupancy.obstacle_height_thresh
        self.obs_occupancy = OccupancyMap(resolution, obstacle_height_thresh)
        self.initial_location = {}
        self.step = -1
        self._metrics = {
            "model_inference_time": 0.0,
            "update_occupancy_time": 0.0,
            "obs_steps": -1,
        }
        self.rgb_images = []
        self.scene_pcd = None
        self.resample_pcd = True
        self.resample_pcd_check_dict = {"z_previous": None, "t_previous": None}

    def render_image(
        self,
        T_cw: np.ndarray,             # 4x4 camera-from-world (OpenCV)
        point_size: float = 3.0,
        bg_rgba=(0, 0, 0, 1),         # RGBA; 0..255 or 0..1
    ):
        """Render color/depth from self.scene_pcd using self.camera intrinsics/near/far."""

        assert self.scene_pcd is not None, "scene_pcd is None"

        # ---- Camera params
        K: np.ndarray = self.camera.intrinsics
        width: int = int(self.camera.w)
        height: int = int(self.camera.h)
        near: float = float(self.camera.near)
        far: float = float(self.camera.far)

        assert K.shape == (3, 3), f"Expected K 3x3, got {K.shape}"
        assert T_cw.shape == (4, 4), f"Expected T_cw 4x4, got {T_cw.shape}"

        # Open3D expects world->camera
        T_wc = np.linalg.inv(T_cw).astype(np.float64, copy=False)  # must be float64 for setup_camera

        # Intrinsics object (used by setup_camera variant #2)
        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width, height,
            float(K[0, 0]), float(K[1, 1]),
            float(K[0, 2]), float(K[1, 2])
        )

        # Offscreen renderer
        renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        scene = renderer.scene

        # Background: expects a single float[4] array (0..1)
        bg = np.asarray(bg_rgba, dtype=np.float32)
        if bg.max() > 1.0:
            bg = bg / 255.0
        scene.set_background(bg)

        # Points material
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        mat.point_size = float(point_size)

        # Geometry
        scene.clear_geometry()
        scene.add_geometry("pcd", self.scene_pcd, mat)

        # Projection + near/far (some builds honor near/far only via set_projection)
        try:
            scene.camera.set_projection(
                intrinsic, near, far,
                o3d.visualization.rendering.Camera.FovType.Unknown
            )
        except Exception:
            pass  # not critical; setup_camera below sets projection via intrinsics

        # ---- Correct setup_camera call (signature #2)
        # (intrinsics: PinholeCameraIntrinsic, extrinsic_matrix: float64 4x4)
        renderer.setup_camera(intrinsic, T_wc)

        # Render RGBA + depth (Z in meters)
        color_img = renderer.render_to_image()
        try:
            depth_img = renderer.render_to_depth_image(z_in_view_space=True)
        except TypeError:
            depth_img = renderer.render_to_depth_image()

        color_np = np.asarray(color_img)                     # (H,W,4) uint8
        depth_np = np.asarray(depth_img, dtype=np.float32)   # (H,W) float32 (meters)

        # Free GPU resources when used in loops
        try:
            renderer.release()
        except Exception:
            pass
        del renderer

        return color_np, depth_np

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
    ) -> List[Any]:
        """Render images of the model given a list of goal camera views."""
        assert len(goals) == len(forwards), "Goals and forwards must have the same length"

        # ensure float
        initial_pose = np.asarray(initial_pose, dtype=np.float64)

        images = []
        for goal, forward in zip(goals, forwards):
            goal = np.asarray(goal, dtype=np.float64).reshape(3)
            forward = BaseWorldModel.normalize(np.asarray(forward, dtype=np.float64).reshape(3))

            # Camera -Z looks along the forward direction
            z_axis = BaseWorldModel.normalize(-forward)
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = BaseWorldModel.normalize(np.cross(y_axis, z_axis))

            R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)   # (3,3)
            t = goal.astype(np.float64).reshape(3, 1)

            T_cam2world = np.eye(4, dtype=np.float64)
            T_cam2world[:3, :3] = R
            T_cam2world[:3, 3:4] = t

            color, _ = self.render_image(
                T_cw=T_cam2world,
            )
            images.append(color)  # (H, W, 3) np array
        render_output = RenderOutput(rgb=images)
        return render_output

    def _build_model(self, model_checkpoint: Optional[str] = None) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        if model_checkpoint is None:
            self.model = VGGT.from_pretrained("facebook/VGGT-1B").to(self.device)
        else:
            self.model = VGGT().to(self.device).eval()
            ckpt = torch.load(model_checkpoint, map_location=self.device)
            if "state_dict" in ckpt:
                self.model.load_state_dict(ckpt["state_dict"], strict=False)
            elif "model" in ckpt:
                self.model.load_state_dict(ckpt["model"], strict=False)
            else:
                self.model.load_state_dict(ckpt, strict=False)
        print("VGGT model loaded on", self.device)

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
            with torch.cuda.amp.autocast(dtype=self.model_dtype):
                images = self._preprocess_images(self.rgb_images)
                images = images[None]  # add batch dimension
                aggregated_tokens_list, ps_idx = self.model.aggregator(images)

            # Predict Cameras
            pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
            extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])  # extrinsic: (1,T,3,4) or (1,T,4,4); intrinsic: (1,T,3,3) or (1,3,3)

            # ---- SCALE: compute depth scale from intrinsics ----
            # K_gt must be a 3x3 (OpenCV) intrinsic for the camera
            K_gt = _as_tensor3x3(self.camera.intrinsics, device=intrinsic.device, dtype=intrinsic.dtype)
            K_est_seq = intrinsic.squeeze(0)  # (T,3,3) or (3,3)
            s_depth = _depth_scale_from_intrinsics(K_est_seq, K_gt)

            # Predict Depth Maps
            depth_map, depth_conf = self.model.depth_head(aggregated_tokens_list, images, ps_idx)

            # ---- scale depths before unprojection ----
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