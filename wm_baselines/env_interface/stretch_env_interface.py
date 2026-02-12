import numpy as np
from omegaconf import DictConfig
from typing import Optional
import timeit
import time
from wm_baselines.task_manager.base_task_manager import BaseTaskManager
from scipy.spatial.transform import Rotation as R
from wm_baselines.env_interface.base_env_interface import BaseEnvInterface
from wm_baselines.utils.planning_utils import distance_traveled_step
from wm_baselines.utils.vision_utils import square_image, pose_robot_to_opencv, pose_opencv_to_robot
from stretch.agent.zmq_client import HomeRobotZmqClient
from stretch.utils.geometry import angle_difference, xyt_base_to_global

class StretchEnvInterface(BaseEnvInterface):
    def __init__(self, embodied_config: DictConfig, task_manager: BaseTaskManager, **kwargs):
        super().__init__(embodied_config, task_manager, **kwargs)
        self.stretch_controller : HomeRobotZmqClient = self.task_manager.stretch_controller
        self.ROTATION_STEP = embodied_config.rotation_step # degrees
        self.MOVE_STEP = embodied_config.move_step # meters
        self.ACTION_MAP = {
            # Navigation actions
            'm': ('MoveAhead', {'moveMagnitude': self.MOVE_STEP}),
            'b': ('MoveBack', {'moveMagnitude': self.MOVE_STEP}),
            'l': ('RotateLeft', {'degrees': self.ROTATION_STEP}),
            'r': ('RotateRight', {'degrees': self.ROTATION_STEP}),
        }
        self.start() # start the robot connection

    @property
    def action_space(self):
        return list(self.ACTION_MAP.keys())
    
    @property
    def image_buffer(self):
        return self._image_buffer

    @property
    def string_buffer(self):
        return self._string_buffer

    def start(
        self,
        verbose: bool = True,
    ) -> None:

        # Call the robot's own startup hooks
        started = self.stretch_controller.start()
        if not started:
            # update here
            raise RuntimeError("Robot failed to start!")

        if verbose:
            print("ZMQ connection to robot started.")

        # First, open the gripper...
        self.stretch_controller.switch_to_manipulation_mode()
        self.stretch_controller.open_gripper()

        # Tuck the arm away
        if verbose:
            print("Sending arm to home...")
        self.stretch_controller.move_to_nav_posture()
        if verbose:
            print("... done.")

        # Move the robot into navigation mode
        self.stretch_controller.switch_to_navigation_mode()

    def reset(self, idx=None):
        self.task_manager.reset(idx)
        super().reset(idx)

    def get_observation(self): # TODO check pose conversion
        obs = self.stretch_controller.get_observation()
        position = np.array([obs.gps[0], obs.gps[1], 0.0])
        rgb = square_image(obs.rgb)
        depth = square_image(obs.depth)
        camera_pose = obs.camera_pose
        # rotate -90 deg around z
        theta = -np.pi / 2.0
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=float)

        A = np.eye(4, dtype=float)
        A[:3, :3] = Rz
        camera_pose = camera_pose @ A
        camera_rotation = camera_pose[:3, :3]
        camera_position = camera_pose[:3, 3]
        observation = {
            "rgb": rgb,
            "depth": depth,
            "pose": camera_pose,
            "position": camera_position,
            "rotation": camera_rotation,
            "robot_position": position,
            "yaw": np.degrees(obs.compass)[0],
        }
        return observation

    ###########
    # Actions #
    ###########

    def turn_left(self):
        action_name, action_args = self.ACTION_MAP["l"]
        x, y, theta = self.stretch_controller.get_base_pose()
        delta_rads = np.radians(action_args["degrees"])
        self.stretch_controller.move_base_to(
            [x, y, theta + delta_rads],
            relative=False,
            blocking=True,
            verbose=True,
        )
        complete = not self.stretch_controller.last_motion_failed()
        self._step_trajectory_logger(action_name, action_args)
        return complete

    def turn_right(self):
        action_name, action_args = self.ACTION_MAP["r"]
        x, y, theta = self.stretch_controller.get_base_pose()
        delta_rads = np.radians(action_args["degrees"])
        self.stretch_controller.move_base_to(
            [x, y, theta - delta_rads],
            relative=False,
            blocking=True,
            verbose=True,
        )
        complete = not self.stretch_controller.last_motion_failed()
        self._step_trajectory_logger(action_name, action_args)
        return complete

    def move_forward(self):
        action_name, action_args = self.ACTION_MAP["m"]
        start = self.stretch_controller.get_base_pose()
        step = action_args["moveMagnitude"]
        forward = np.array([step, 0, 0])
        xyt_goal_forward = xyt_base_to_global(forward, start)
        self.stretch_controller.move_base_to(
            xyt_goal_forward,
            relative=False,
            blocking=True,
            verbose=True,
        )
        complete = not self.stretch_controller.last_motion_failed()
        self._step_trajectory_logger(action_name, action_args)
        return complete
    
    def move_back(self):
        action_name, action_args = self.ACTION_MAP["b"]
        start = self.stretch_controller.get_base_pose()
        step = action_args["moveMagnitude"]
        backward = np.array([-1 * step, 0, 0])
        xyt_goal_backward = xyt_base_to_global(backward, start)
        self.stretch_controller.move_base_to(
            xyt_goal_backward,
            relative=False,
            blocking=True,
            verbose=True,
        )
        complete = not self.stretch_controller.last_motion_failed()
        self._step_trajectory_logger(action_name, action_args)
        return complete

    def nav_to(self, target_pose: np.ndarray, max_time: float = 10.0):
        """
        Navigate to a target pose.
        """
        # import ipdb; ipdb.set_trace()
        theta = R.from_matrix(target_pose[:3, :3]).as_euler("xyz", degrees=False)[1]
        start = self.stretch_controller.get_base_pose()
        x = target_pose[0, 3]
        y = target_pose[1, 3]
        goal = [x, y, start[2]]
        self.stretch_controller.move_base_to(
            goal, 
            relative=False,
            blocking=True,
            verbose=True
        )
        # start = self.stretch_controller.get_base_pose()
        # self.stretch_controller.move_base_to(
        #     [start[0], start[1], theta],
        #     relative=False,
        #     blocking=True,
        #     verbose=True,
        # )
        # half chance turn left or right first
        # set seed using current time in ns
        # np.random.seed(int(time.time_ns() % (2**32 - 1)))
        # if np.random.rand() < 0.5:
        #     self.turn_right()
        #     self.turn_right()
        # else:
        #     self.turn_left()
        #     self.turn_left()
        self.turn_left()
        self.turn_left()
        complete = not self.stretch_controller.last_motion_failed()
        self._step_trajectory_logger("nav_to", {"target_pose": target_pose})
        return complete
        
    def end(self):
        self.stretch_controller.stop()

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
