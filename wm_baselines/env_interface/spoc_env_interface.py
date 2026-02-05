import numpy as np
from omegaconf import DictConfig, OmegaConf
from environment.stretch_controller import StretchController
from belief_baselines.task_manager.base_task_manager import BaseTaskManager
from scipy.spatial.transform import Rotation as R
from spoc_utils.embodied_utils import square_image, distance_traveled_step
from belief_baselines.env_interface.base_env_interface import BaseEnvInterface


class SpocEnvInterface(BaseEnvInterface):
    def __init__(self, embodied_config: DictConfig, task_manager: BaseTaskManager, **kwargs):
        super().__init__(embodied_config, task_manager, **kwargs)
        self.stretch_controller : StretchController = self.task_manager.stretch_controller
        self.ROTATION_STEP = embodied_config.rotation_step # degrees
        self.MOVE_STEP = embodied_config.move_step # meters
        self.ACTION_MAP = {
            # Navigation actions
            'm': ('MoveAhead', {'moveMagnitude': self.MOVE_STEP}),
            'b': ('MoveBack', {'moveMagnitude': self.MOVE_STEP}),
            'l': ('RotateLeft', {'degrees': self.ROTATION_STEP}),
            'r': ('RotateRight', {'degrees': self.ROTATION_STEP}),
        }

    @property
    def action_space(self):
        return list(self.ACTION_MAP.keys())
    
    @property
    def image_buffer(self):
        return self._image_buffer

    @property
    def string_buffer(self):
        return self._string_buffer

    def reset(self, idx=None):
        self.task_manager.reset(idx)
        super().reset(idx)

    def get_observation(self): # TODO check posse conversion
        # get RGB
        rgb_image = square_image(self.stretch_controller.navigation_camera)
        # get depth
        depth_image = square_image(self.stretch_controller.navigation_depth_frame)
        # get camera pose in world coordinates
        event = self.stretch_controller.controller.last_event
        pos = event.metadata["cameraPosition"]                      # dict x,y,z (world)
        agent = event.metadata["agent"]
        yaw = agent["rotation"]["y"]                 # degrees (around Y)
        pitch = agent["cameraHorizon"]

        R_aw = R.from_euler('xyz', [0.0, yaw, 0.0], degrees=True).as_matrix() # R_aw * R_ca = R_cw
        R_ca = R.from_euler('xyz', [pitch, 0.0, 0.0], degrees=True).as_matrix()
        R_cw = R_aw @ R_ca
        t_cw = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
        pose = np.eye(4, dtype=float)
        pose[:3,:3] = R_cw
        pose[:3, 3] = t_cw
        pose = self._lh2rh(pose)

        observation = {
            "rgb": rgb_image,
            "depth": depth_image,
            "pose": pose,
            "position": t_cw,
            "rotation": R_cw,
            "yaw": yaw,
        }
        return observation

    ###########
    # Actions #
    ###########

    def turn_left(self):
        action_name, action_args = self.ACTION_MAP["l"]
        complete = self.stretch_controller.controller.step(action=action_name, **action_args).metadata["lastActionSuccess"]
        self._step_trajectory_logger(action_name, action_args)
        return complete

    def turn_right(self):
        action_name, action_args = self.ACTION_MAP["r"]
        complete = self.stretch_controller.controller.step(action=action_name, **action_args).metadata["lastActionSuccess"]
        self._step_trajectory_logger(action_name, action_args)
        return complete

    def move_forward(self):
        action_name, action_args = self.ACTION_MAP["m"]
        complete = self.stretch_controller.controller.step(action=action_name, **action_args).metadata["lastActionSuccess"]
        self._step_trajectory_logger(action_name, action_args)
        return complete
    
    def move_back(self):
        action_name, action_args = self.ACTION_MAP["b"]
        complete = self.stretch_controller.controller.step(action=action_name, **action_args).metadata["lastActionSuccess"]
        self._step_trajectory_logger(action_name, action_args)
        return complete
    
    def nav_to(self, target_pose: np.ndarray, max_steps_per_segment: int = 50):
        """
        Navigate with per-segment step caps. Breaks out of each while-loop if the
        number of steps reaches `max_steps_per_segment`.
        """
        complete = True
        target_pose = self._rh2lh(target_pose)

        print("Navigating to:", target_pose)

        # == pick closest reachable ==
        target_position = target_pose[:3, 3]
        reachable_positions = self.stretch_controller.controller.step(
            action="GetReachablePositions"
        ).metadata["actionReturn"]
        reachable_positions = np.array(
            [[p["x"], p["y"], p["z"]] for p in reachable_positions], dtype=float
        )
        dists = np.linalg.norm(reachable_positions - target_position[None, :], axis=1)
        closest_position = reachable_positions[np.argmin(dists), :]

        # == compute face-to yaw ==
        current_event = self.stretch_controller.controller.last_event
        current_agent = current_event.metadata["agent"]
        current_yaw = current_agent["rotation"]["y"]  # degrees
        current_observation = self.get_observation()
        current_position = current_observation["position"]
        current_forward = current_observation["rotation"][:, 2]  # (3,)
        to_target = closest_position - current_position
        to_target[1] = 0.0
        nrm = np.linalg.norm(to_target)
        if nrm < 1e-9:
            to_target = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            to_target = to_target / nrm

        dot = float(np.clip(np.dot(current_forward, to_target), -1.0, 1.0))
        angle = np.arccos(dot)  # [0,pi]
        cross = np.cross(current_forward, to_target)
        if cross[1] < 0:
            angle = -angle
        angle = np.degrees(angle)
        face_to_yaw = (current_yaw + angle) % 360

        # == rotate to face closest position ==
        steps = 0
        yaw_diff = (face_to_yaw - current_yaw + 180) % 360 - 180  # [-180,180]
        while abs(yaw_diff) > self.ROTATION_STEP:
            complete = False
            print(f"Rotating: {yaw_diff} degrees left")
            if steps >= max_steps_per_segment:
                break
            if yaw_diff > 0:
                action_name = "RotateRight"
                action_args = self.ACTION_MAP["r"][1]
                self.stretch_controller.controller.step(action=action_name, **action_args)
            else:
                action_name = "RotateLeft"
                action_args = self.ACTION_MAP["l"][1]
                self.stretch_controller.controller.step(action=action_name, **action_args)

            current_event = self.stretch_controller.controller.last_event
            current_agent = current_event.metadata["agent"]
            current_yaw = current_agent["rotation"]["y"]
            yaw_diff = (face_to_yaw - current_yaw + 180) % 360 - 180
            self._step_trajectory_logger(action_name, action_args)
            steps += 1
            break

        # == move to closest position ==
        current_position = np.array(
            [
                current_agent["position"]["x"],
                current_agent["position"]["y"],
                current_agent["position"]["z"],
            ],
            dtype=float,
        )

        steps = 0
        while complete and np.linalg.norm(current_position - closest_position) > self.MOVE_STEP:
            complete = False
            print(f"Moving: {np.linalg.norm(current_position - closest_position)} m left")
            if steps >= max_steps_per_segment:
                break

            self.stretch_controller.controller.step(
                action="MoveAhead", **self.ACTION_MAP["m"][1]
            )
            current_event = self.stretch_controller.controller.last_event
            current_agent = current_event.metadata["agent"]
            current_position = np.array(
                [
                    current_agent["position"]["x"],
                    current_agent["position"]["y"],
                    current_agent["position"]["z"],
                ],
                dtype=float,
            )
            self._step_trajectory_logger("MoveAhead", self.ACTION_MAP["m"][1])
            steps += 1
            break
        return complete

    def end(self):
        self.stretch_controller.controller.step(action="EndEpisode")

    ####################
    # Internal Methods #
    ####################
    def _step_trajectory_logger(self, action_name, action_args):
        distance_traveled, angle_turned = distance_traveled_step(action_name, action_args)
        self.task_manager._distance_traveled += distance_traveled
        self.task_manager._angle_turned += angle_turned
        self.task_manager._current_step += 1
        if self.save_trajectory:
            obs = self.get_observation()
            image_dict = {key: obs[key] for key in self.trajectory_image_keys if key in obs}
            string_dict = {key: obs[key] for key in self.trajectory_string_keys if key in obs}
            if 'action' in self.trajectory_string_keys:
                string_dict['action'] = (action_name, action_args)
            if 'distance_to_goal' in self.trajectory_string_keys:
                string_dict['distance_to_goal'] = self.task_manager.distance_to_goal
            if 'distance_traveled' in self.trajectory_string_keys:
                string_dict['distance_traveled'] = self.task_manager.distance_traveled
            if 'angle_turned' in self.trajectory_string_keys:
                string_dict['angle_turned'] = self.task_manager.angle_turned
            for key, value in image_dict.items():
                self._image_buffer[key].append(value)
            for key, value in string_dict.items():
                self._string_buffer[key].append(value.tolist() if isinstance(value, np.ndarray) else value)

    def _lh2rh(self, pose):
        """
        Convert left-handed to right-handed coordinate system.
        """
        F = np.diag([1, 1, -1, 1])
        pose =  F @ pose @ F
        pose = np.linalg.inv(pose)
        conversion = np.diag([1, -1, -1, 1])
        pose = conversion @ pose
        pose = np.linalg.inv(pose)
        return pose

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