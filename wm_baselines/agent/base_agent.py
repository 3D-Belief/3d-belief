from typing import Any, Dict, List, Optional, Union
import numpy as np
#import open3d as o3d
from PIL import Image
import torch
from torch import Tensor
from pathlib import Path
from wm_baselines.env_interface.base_env_interface import BaseEnvInterface
from wm_baselines.world_model.base_world_model import BaseWorldModel
from wm_baselines.planner.base_planner import BasePlanner
from wm_baselines.agent.perception.camera import Camera

class BaseAgent:
    """Base agent class providing a minimal interface.

    Responsibilities:
    - define an act() method to be implemented by subclasses
    """

    def __init__(self,
                 env_interface: BaseEnvInterface, 
                 world_model: BaseWorldModel, 
                 planner: BasePlanner,
                 camera: Optional[Camera] = None,
                 assets_saver: Optional[Dict[str, Any]] = None,
                 **kwargs
        ):
        self.env_interface = env_interface
        self.world_model = world_model
        self.planner = planner
        self.camera = camera
        self.save_assets = assets_saver and assets_saver.get("save", False)
        self.assets_save_path = assets_saver.get("save_path", None) if assets_saver else None
        if self.save_assets and self.assets_save_path:
            self.assets_save_path = Path(self.assets_save_path)
            self.assets_save_path.mkdir(parents=True, exist_ok=True)
        self.assets_save_keys = assets_saver.get("asset_keys", []) if assets_saver else []
        self.assets_save_path_ep = None
        self.current_ep_name = None

        # Current state
        self.current_observation: Dict[str, Any] = None
        self.current_assets: Dict[str, Any] = None
        self.step = -1
        self.force_update = False

    def reset(self):
        self.current_observation = None
        self.current_assets = None
        self.step = -1
        self.force_update = False
        self.current_ep_name = None

    def observe(self) -> Dict[str, Any]: #  complete
        """Get the current observation and return the observation dict."""
        obs = self.env_interface.get_observation()
        assets = self.world_model.update_observation(obs, self.force_update)
        self.planner.set_state(assets)
        self.current_observation = obs
        self.current_assets = assets
        self.step += 1
        return obs

    def plan(self) -> Dict[str, Any]:
        """Plan the next action. """
        obs = self.env_interface.get_observation()
        pose = obs.get("pose")
        if self.world_model.initial_location.get('pose') is not None:
            pose_map = np.linalg.inv(self.world_model.initial_location['pose']) @ pose
            self.current_assets['pose'] = pose_map
        action, trace_asset = self.planner.get_next_action(self.current_assets, self.step, self.force_update)
        self._step_assets_save(trace_asset)
        return action

    def act(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Given an action, invoke env_interface and return the resulting observation dict. Must be implemented by subclass."""
        action_fn = getattr(self.env_interface, action["action_name"])
        complete = action_fn(**action["args"])
        self.force_update = True if complete else False

    def render_image(
        self,
        extrinsics: Union[np.ndarray, Tensor] = None,              # (4, 4)
        intrinsics: Optional[Union[np.ndarray, Tensor]] = None,
        near: Optional[Union[float, Tensor]] = None,
        far: Optional[Union[float, Tensor]] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> Any:
        """Render image of the internal scene given a camera view."""
        assert self.world_model is not None, "no 3D world model exist in this agent"
        if intrinsics is None:
            intrinsics = torch.tensor(self.camera.intrinsics)
        if near is None:
            near = torch.tensor(self.camera.near)
        if far is None:
            far = torch.tensor(self.camera.far)
        if h is None or w is None:
            h, w = self.camera.h, self.camera.w
        output = self.world_model.render_image( # TODO 
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            near=near,
            far=far,
            h=h,
            w=w,
        )
        return output

    def render_video(
        self,
        render_poses: List[Union[np.ndarray, Tensor]],              # List of (4, 4)
        intrinsics: Optional[Union[np.ndarray, Tensor]] = None,
        near: Optional[Union[float, Tensor]] = None,
        far: Optional[Union[float, Tensor]] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> Any:
        """Render video of the internal scene given a camera trajectory."""
        assert self.world_model is not None, "no 3D world model exist in this agent"
        if intrinsics is None:
            intrinsics = torch.tensor(self.camera.intrinsics)
        if near is None:
            near = torch.tensor(self.camera.near)
        if far is None:
            far = torch.tensor(self.camera.far)
        if h is None or w is None:
            h, w = self.camera.h, self.camera.w
        output = self.world_model.render_video( # TODO 
            render_poses=render_poses,
            intrinsics=intrinsics,
            near=near,
            far=far,
            h=h,
            w=w,
        )
        return output

    def export_scene(self, path: Path, extrinsics: Union[Tensor, np.ndarray]):
        """Export the internal scene to a mesh file."""
        assert self.world_model is not None, "no 3D world model exist in this agent"
        self.world_model.export_scene(path, extrinsics)

    def _setup_assets_save(self) -> None:
        """Setup assets saving configuration."""
        if self.save_assets and self.assets_save_path:
            self.assets_save_path = Path(self.assets_save_path)
            self.assets_save_path_ep = self.assets_save_path / self.current_ep_name / "trace"
            self.assets_save_path_ep.mkdir(parents=True, exist_ok=True)
            for key in self.assets_save_keys:
                (self.assets_save_path_ep / key).mkdir(parents=True, exist_ok=True)

    def _step_assets_save(self, trace_asset: Dict[str, Any]) -> None:
        """Return the current step's assets for saving/logging."""
        raise NotImplementedError()
