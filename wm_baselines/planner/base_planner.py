from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
from torch import Tensor
from pathlib import Path
from omegaconf import DictConfig

class BasePlanner:
    """Base planner class providing a minimal interface.

    Responsibilities:
    - hold config reference
    - define a plan() method to be implemented by subclasses
    """

    def __init__(self, **kwargs):
        self.current_asset: Optional[Dict[str, Any]] = None  # hold current asset
        self._metrics : Dict[str, float] = {}  # time taken for inference in seconds
        self.step = -1

    @property
    def metrics(self) -> Dict[str, float]:
        return self._metrics
    
    def reset(self):
        """Reset the planner state if needed. Subclasses can override."""
        self.current_asset = None
        self._metrics = {}
        self.step = -1
    
    def set_state(self, current_asset: Dict[str, Any] = None) -> None:
        """Set the current state for the planner."""
        # Update internal state based on current observation and assets if needed
        self.current_asset = current_asset
        self.step += 1

    def get_next_action(self, assets: Dict[str, Any]) -> Dict[str, Any]:
        """Get the next action. Must be implemented by subclass."""
        raise NotImplementedError()

    def plan(self) -> Dict[str, Any]:
        """Given the current assets, return the next action dict. Must be implemented by subclass."""
        raise NotImplementedError()