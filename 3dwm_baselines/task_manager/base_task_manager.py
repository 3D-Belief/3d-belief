from typing import Optional
import time


class BaseTaskManager:
    """Base task manager providing common metrics and a minimal reset contract.

    Subclasses should implement their own reset logic and call super().reset().
    """

    def __init__(self, embodied_config=None, **kwargs):
        self.embodied_config = embodied_config
        self.episodes = []
        self._distance_traveled: float = 0.0
        self._angle_turned: float = 0.0
        self._current_step: int = 0
        self.start_time: Optional[float] = None

    @property
    def distance_traveled(self) -> float:
        return self._distance_traveled

    @property
    def angle_turned(self) -> float:
        return self._angle_turned

    @property
    def current_step(self) -> int:
        return self._current_step

    def reset(self, idx: Optional[int] = None):
        self._distance_traveled = 0.0
        self._angle_turned = 0.0
        self._current_step = 0
        self.start_time = time.time()

    def is_done(self):
        pass

    def get_final_log(self, metrics: dict) -> dict:
        pass