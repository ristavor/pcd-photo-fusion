# src/synchronizer/synchronizer.py
from pathlib import Path
from typing import List, Tuple

import numpy as np
from functools import cached_property

from .loader import TimestampLoader
from .matcher import TimeMatcher
from utils.constants import DEFAULT_THRESHOLD_FACTOR

class Synchronizer:
    """
    Фасад для синхронизации raw-данных камеры и LiDAR-а по timestamps.
    """

    def __init__(self, raw_root: Path, cam_folder: str, threshold: float = None):
        self.raw_root = Path(raw_root)
        self.cam_folder = cam_folder
        self._user_threshold = threshold

    @cached_property
    def loader(self) -> TimestampLoader:
        return TimestampLoader(self.raw_root, self.cam_folder)

    @cached_property
    def cam_times(self) -> list[float]:
        return self.loader.camera_timestamps

    @cached_property
    def velo_times(self) -> list[float]:
        return self.loader.velo_timestamps

    @cached_property
    def threshold(self) -> float:
        if self._user_threshold is not None:
            return self._user_threshold
        dt_cam  = np.diff(self.cam_times)
        dt_velo = np.diff(self.velo_times)
        return DEFAULT_THRESHOLD_FACTOR * min(float(np.median(dt_cam)), float(np.median(dt_velo)))

    @cached_property
    def matcher(self) -> TimeMatcher:
        return TimeMatcher(self.threshold)

    def match_pairs(self) -> List[Tuple[int, int]]:
        """
        Возвращает список (cam_idx, velo_idx) для тех пар, где |Δt| ≤ threshold.
        """
        return self.matcher.match_pairs(self.cam_times, self.velo_times)
