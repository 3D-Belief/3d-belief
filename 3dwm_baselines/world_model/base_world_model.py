from typing import Any, Dict, List, Optional, Union
import numpy as np
from omegaconf import DictConfig
from belief_baselines.agent.perception.occupancy import OccupancyMap

class BaseWorldModel:
    """Base world model class providing a minimal interface.

    Responsibilities:
    - hold config reference
    - define a predict() method to be implemented by subclasses
    """

    def __init__(self, **kwargs):
        self.obs_occupancy : Optional[OccupancyMap] = None  # optionally maintain an occupancy map
        self.belief_occupancy : Optional[OccupancyMap] = None  # optionally maintain a belief occupancy map
        self._metrics : Dict[str, float] = {}  # time taken for inference in seconds

    @property
    def metrics(self) -> Dict[str, float]:
        return self._metrics

    def reset(self):
        """Reset the world model state if needed. Subclasses can override."""
        self.obs_occupancy = None
        self.belief_occupancy = None
        self._metrics = {}

    def update_observation(self, observation: Dict[str, Any], force_update: bool) -> Dict[str, Any]:
        """Update the world model with a new observation."""
        rgb = observation.get("rgb")
        depth = observation.get("depth")
        position = observation.get("position")  # (3,)
        rotation = observation.get("rotation")  # (3, 3)
        yaw = observation.get("yaw")  # float, in radians
        assets = {
            "rgb": rgb,
            "depth": depth,
            "position": position,
            "rotation": rotation,
            "yaw": yaw,
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
    ) -> Any:
        """Render image of the internal scene given a camera view. Must be implemented by subclass."""
        raise NotImplementedError()

    def render_video(
        self,
        extrinsics: List[Any],              # List of (4, 4)
        intrinsics: Optional[List[Any]] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> Any:
        """Render video of the internal scene given a list of camera views. Must be implemented by subclass."""
        raise NotImplementedError()
    
    def export_scene(self, save_path: str) -> None:
        """Export the internal scene to a file. Must be implemented by subclass."""
        raise NotImplementedError()

    def render_goal_images(
        self,
        goals: List[np.ndarray],           # List of (3, 3) camera positions
        forwards: List[np.ndarray],        # List of (3,) forward vectors
        intrinsics: Optional[Any] = None,
        near: Optional[float] = None,
        far: Optional[float] = None,
        h: Optional[int] = None,
        w: Optional[int] = None,
    ) -> List[Any]:
        """Render images at the goal locations. Must be implemented by subclass."""
        raise NotImplementedError()
    
    @staticmethod
    def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        n = float(np.linalg.norm(v))
        return v if n < eps else (v / n)