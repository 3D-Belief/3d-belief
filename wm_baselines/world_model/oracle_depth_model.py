from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig
import numpy as np
import open3d as o3d
import torch
from torch import Tensor
from belief_baselines.utils.data_classes import Frame, RenderOutput
from belief_baselines.agent.perception.camera import Camera
from belief_baselines.agent.perception.occupancy import OccupancyMap
from belief_baselines.agent.perception.reconstruction import TSDFFusion
from belief_baselines.world_model.base_world_model import BaseWorldModel
from belief_baselines.utils.vision_utils import pose_lh2rh, pose_gl_cam2world_to_open3d_cam2world
from belief_baselines.utils.planning_utils import rotation_angle

class OracleDepthModel(BaseWorldModel):
    """Oracle depth world model using ground truth depth from the environment.

    Responsibilities:
    - hold config reference
    - maintain an occupancy map using ground truth depth
    - implement update_observation() to update the occupancy map
    - implement render_image() and render_video() to visualize the occupancy map
    """

    def __init__(
        self, 
        model: TSDFFusion, 
        obs_occupancy: OccupancyMap, 
        camera: Camera,
        adjacent_angle: float,
        adjacent_distance: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model = model
        self.obs_occupancy = obs_occupancy
        self.camera = camera
        self.initial_location = {}
        self.resample_pcd = True
        self.resample_pcd_check_dict = {}
        self.adjacent_angle = adjacent_angle
        self.adjacent_distance = adjacent_distance
        self.step = -1
        self._metrics = {
            "model_inference_time": 0.0,
            "update_occupancy_time": 0.0,
            "obs_steps": -1,
        }

    def reset(self):
        """Reset the occupancy maps."""
        resolution = self.obs_occupancy.resolution
        obstacle_height_thresh = self.obs_occupancy.obstacle_height_thresh
        self.obs_occupancy = OccupancyMap(resolution, obstacle_height_thresh)
        self.model.reset()
        self.initial_location = {}
        self.resample_pcd = True
        self.resample_pcd_check_dict = {}
        self.step = -1
        self._metrics = {
            "model_inference_time": 0.0,
            "update_occupancy_time": 0.0,
            "obs_steps": -1,
        }

    def update_observation(self, observation: Dict[str, Any], force_update: bool) -> Dict[str, Any]:
        """Update the occupancy maps with ground truth depth."""
        self.step += 1
        self._metrics["obs_steps"] += 1
        depth = observation.get("depth")
        rgb = observation.get("rgb")
        pose = observation.get("pose")  # (4, 4) world to camera
        position = observation.get("position")
        rotation = observation.get("rotation")

        if self.initial_location.get('pose') is None:
            self.initial_location['pose'] = pose
            self.initial_location['position'] = position
            self.initial_location['rotation'] = rotation
        
        if self.initial_location.get('pose') is not None:
            pose_map = np.linalg.inv(self.initial_location['pose']) @ pose # make the initial pose the origin
            position = pose_map[:3, 3]
            rotation = pose_map[:3, :3]

        self._update_resample_pcd(pose, force=force_update)
        if self.resample_pcd:
            self.resample_pcd = False
            pcd = OracleDepthModel.depth_to_pcd(
                depth, self.camera.fx, self.camera.fy, self.camera.cx, self.camera.cy,
                rgb=rgb, T_cam2world=pose, invalid_val=0
            )
            _, exe_time = self.model.integrate_frame(depth=depth, color=rgb, T_cam2world=pose)
            self._metrics["model_inference_time"] += exe_time

            pcd = pcd.transform(np.linalg.inv(self.initial_location['pose']))

            _, exe_time = self.obs_occupancy.integrate(np.array(pcd.points), position, rotation, intrinsics=self.camera.intrinsics)
            self._metrics["update_occupancy_time"] += exe_time

        assets = {
            "rgb": rgb,
            "depth": depth,
            "occupancy": self.obs_occupancy,
            # "pcd": self.model.extract_point_cloud(),
            "pose": pose_map,
            "initial_pose": self.initial_location['pose'],
        }
        return assets

    def render_image(
        self,
        extrinsics: Any,           # (4, 4)
        intrinsics: Optional[Any] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> Frame:
        """Render image of the model given a camera view."""
        return self.model.render_view(T_cam2world=extrinsics, width=w, height=h,
                                      min_depth=near, max_depth=far,
                                      depth_scale_out=1.0, return_normals=False)

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
            forward = OracleDepthModel.normalize(np.asarray(forward, dtype=np.float64).reshape(3))

            # Camera -Z looks along the forward direction
            z_axis = OracleDepthModel.normalize(-forward)
            y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            x_axis = OracleDepthModel.normalize(np.cross(y_axis, z_axis))

            R = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float64)   # (3,3)
            t = goal.astype(np.float64).reshape(3, 1)

            T_cam2world = np.eye(4, dtype=np.float64)
            T_cam2world[:3, :3] = R
            T_cam2world[:3, 3:4] = t

            T_cam2world = initial_pose @ T_cam2world

            frame = self.render_image(
                extrinsics=T_cam2world,
                intrinsics=intrinsics,
                near=near,
                far=far,
                h=h,
                w=w,
            )
            images.append(frame.rgb)  # (H, W, 3) np array
        render_output = RenderOutput(rgb=images)
        return render_output

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

    @staticmethod
    def depth_to_pcd(depth_m, fx, fy, cx, cy, rgb=None, T_cam2world=None, invalid_val=0) -> o3d.geometry.PointCloud:
        """
        depth_m:  (H, W) depth in meters
        rgb:      (H, W, 3) uint8 or None
        T_cam2world: (4,4) or None
        """
        H, W = depth_m.shape
        u = np.arange(W)
        v = np.arange(H)
        uu, vv = np.meshgrid(u, v)

        Z = depth_m
        valid = (Z > 0) & np.isfinite(Z) & (Z != invalid_val)

        X = (uu - cx) * Z / fx
        Y = (vv - cy) * Z / fy

        pts = np.stack((X[valid], Y[valid], Z[valid]), axis=1)  # (N,3)

        if T_cam2world is not None:
            pts_h = np.c_[pts, np.ones(len(pts))]
            pts = (pts_h @ T_cam2world.T)[:, :3]

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

        if rgb is not None:
            colors = (rgb.reshape(-1, 3)[valid.reshape(-1)] / 255.0).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd
    
    @staticmethod
    def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        n = float(np.linalg.norm(v))
        return v if n < eps else (v / n)