from dataclasses import dataclass
from typing import Any, Optional, List
from torch import Tensor

@dataclass
class Frame:
    rgb: Any                   # (H, W, 3) np.uint8
    depth: Any                 # (H, W) np.float32 in meters
    normals: Optional[Any]     # (H, W, 3) np.float32 or None

@dataclass
class RenderOutput:
    rgb: List[Tensor]
    depth: Optional[List[Tensor]] = None
    semantic: Optional[List[Tensor]] = None
    pose: Optional[List[Tensor]] = None
