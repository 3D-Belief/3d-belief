from typing import Any, Dict, List, Optional, Union
from omegaconf import DictConfig
import numpy as np
import open3d as o3d
import torch
from torch import Tensor
from belief_baselines.utils.common_utils import with_timing
from belief_baselines.utils.data_classes import Frame, RenderOutput
from belief_baselines.agent.perception.camera import Camera
from belief_baselines.agent.perception.occupancy import OccupancyMap
from belief_baselines.agent.perception.reconstruction import TSDFFusion
from belief_baselines.world_model.oracle_depth_model import OracleDepthModel
from belief_baselines.utils.vision_utils import pose_lh2rh, pose_gl_cam2world_to_open3d_cam2world
from belief_baselines.utils.planning_utils import rotation_angle
from environment.stretch_controller import StretchController
from spoc_utils.embodied_utils import square_image

class OracleImaginationModel(OracleDepthModel):
    """Oracle world model using ground truth depth and simulation renders for imagination.

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
        belief_occupancy: OccupancyMap,
        camera: Camera,
        adjacent_angle: float,
        adjacent_distance: float,
        **kwargs
    ):
        super().__init__(
            model=model, 
            obs_occupancy=obs_occupancy, 
            camera=camera,
            adjacent_angle=adjacent_angle,
            adjacent_distance=adjacent_distance,
            **kwargs
        )
        self.simulation = None
        self.belief_occupancy = belief_occupancy
        self.step = -1
        self._metrics = {
            "model_inference_time": 0.0,
            "update_occupancy_time": 0.0,
            "model_inference_time_imagine": 0.0,
            "obs_steps": -1,
        }
    
    def set_simulation(self, simulation: Any) -> None:
        """Set the simulation environment for rendering imagined views."""
        if not isinstance(simulation, StretchController):
            raise ValueError("Only StretchController instances are supported for simulation.")
        self.simulation = simulation

    def reset(self):
        """Reset the occupancy maps."""
        resolution = self.obs_occupancy.resolution
        obstacle_height_thresh = self.obs_occupancy.obstacle_height_thresh
        self.obs_occupancy = OccupancyMap(resolution, obstacle_height_thresh)
        resolution = self.belief_occupancy.resolution
        obstacle_height_thresh = self.belief_occupancy.obstacle_height_thresh
        self.belief_occupancy = OccupancyMap(resolution, obstacle_height_thresh)
        self.model.reset()
        self.initial_location = {}
        self.resample_pcd = True
        self.resample_pcd_check_dict = {}
        self.step = -1
        self._metrics = {
            "model_inference_time": 0.0,
            "update_occupancy_time": 0.0,
            "model_inference_time_imagine": 0.0,
            "obs_steps": -1,
        }

    def _rh2lh(self, pose):
        """
        Convert right-handed to left-handed coordinate system.
        """
        pose = np.linalg.inv(pose)
        conversion = np.diag([1, -1, -1, 1])
        pose = conversion @ pose
        pose = np.linalg.inv(pose)
        F = np.diag([1, 1, -1, 1])
        pose =  F @ pose @ F
        return pose

    @with_timing
    def _imagine_with_simulation(
        self, 
        imagine_goal_pose: np.ndarray, 
    ) -> Any:
        """Imagine in place given a goal pose."""
        if self.simulation is None:
            raise ValueError("Simulation environment is not set for imagination.")
        target_pose = self._rh2lh(imagine_goal_pose)
        # pick closest reachable
        target_position = target_pose[:3, 3]
        reachable_positions = self.simulation.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        reachable_positions = np.array(
            [[p["x"], p["y"], p["z"]] for p in reachable_positions], dtype=float
        )
        dists = np.linalg.norm(reachable_positions - target_position[None, :], axis=1)
        closest_position = reachable_positions[np.argmin(dists), :]
        # calculate yaw angle
        R_cw = target_pose[:3, :3]
        yaw_rad = np.arctan2(-R_cw[2, 0], R_cw[0, 0])   # yaw about +Y
        yaw_deg = np.degrees(yaw_rad)
        yaw = (yaw_deg + 180) % 360 - 180
        # save the current position and rotation for restoration later
        event = self.simulation.controller.last_event
        agent = event.metadata["agent"]
        current_position = agent["position"]
        current_rotation = agent["rotation"]        
        # teleport the agent to the goal pose
        self.simulation.controller.step(
            action="Teleport",
            position={"x": float(closest_position[0]), "y": float(closest_position[1]), "z": float(closest_position[2])},
            rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
        )
        # render from the agent's camera
        rgb_image = square_image(self.simulation.navigation_camera)
        # restore the agent's position and rotation
        self.simulation.controller.step(
            action="Teleport",
            position={"x": float(current_position["x"]), "y": float(current_position["y"]), "z": float(current_position["z"])},
            rotation={"x": 0.0, "y": float(current_rotation["y"]), "z": 0.0},
        )

        return rgb_image

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

            frame, exe_time = self._imagine_with_simulation(
                imagine_goal_pose=T_cam2world,
            )
            images.append(frame)  # (H, W, 3) np array
            self._metrics["model_inference_time_imagine"] += exe_time

        render_output = RenderOutput(rgb=images)

        return render_output
