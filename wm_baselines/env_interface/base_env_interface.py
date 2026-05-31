from typing import Any, Dict, List, Optional
from omegaconf import DictConfig
from wm_baselines.task_manager.base_task_manager import BaseTaskManager


class BaseEnvInterface:
    """Minimal environment interface base class.

    Responsibilities:
    - hold config reference
    - manage optional trajectory image/string buffers
    - define an action_space property and basic contract methods to implement in subclasses
    """

    def __init__(self, embodied_config: DictConfig, task_manager: BaseTaskManager, **kwargs):
        self.embodied_config = embodied_config
        self.task_manager = task_manager
        self.save_trajectory = getattr(embodied_config, "trajectory", None) and getattr(embodied_config.trajectory, "save", False)
        self.trajectory_save_path = getattr(embodied_config.trajectory, "save_path", None) if getattr(embodied_config, "trajectory", None) else None
        self.trajectory_image_keys: Optional[List[str]] = list(getattr(embodied_config.trajectory, "image_keys", [])) if getattr(embodied_config, "trajectory", None) else []
        self.trajectory_string_keys: Optional[List[str]] = list(getattr(embodied_config.trajectory, "string_keys", [])) if getattr(embodied_config, "trajectory", None) else []
        self._image_buffer: Optional[Dict[str, List[Any]]] = {key: [] for key in self.trajectory_image_keys} if self.save_trajectory else None
        self._string_buffer: Optional[Dict[str, List[Any]]] = {key: [] for key in self.trajectory_string_keys} if self.save_trajectory else None

    @property
    def action_space(self):
        """Return list of available action keys. Subclasses should override if needed."""
        raise NotImplementedError()

    @property
    def image_buffer(self) -> Optional[Dict[str, List[Any]]]:
        return self._image_buffer

    @property
    def string_buffer(self) -> Optional[Dict[str, List[Any]]]:
        return self._string_buffer

    def reset(self, idx: Optional[int] = None) -> None:
        """Reset the buffers; subclass should call super().reset(...) if overriding."""
        self._image_buffer = {key: [] for key in self.trajectory_image_keys} if self.save_trajectory else None
        self._string_buffer = {key: [] for key in self.trajectory_string_keys} if self.save_trajectory else None

    def get_observation(self):
        """Return observation dict. Must be implemented by subclass."""
        raise NotImplementedError()

    def nav_to(self, target_pose):
        """Navigate to target pose. Must be implemented by subclass if navigation is supported."""
        raise NotImplementedError()

