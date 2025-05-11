from functools import cached_property
from pathlib import Path
from typing import List, Tuple

import numpy as np

from utils.constants import DEFAULT_THRESHOLD_FACTOR
from .loader import TimestampLoader
from .matcher import TimeMatcher


class Synchronizer:
    """
    Фасад: загружает timestamps, вычисляет threshold и синхронизирует пары.
    """

    def __init__(self, raw_root: Path, cam_folder: str, threshold: float = None):
        self.raw_root = Path(raw_root)
        self.cam_folder = cam_folder
        self._user_threshold = threshold

    @cached_property
    def loader(self) -> TimestampLoader:
        return TimestampLoader(self.raw_root, self.cam_folder)

    @cached_property
    def cam_times(self) -> List[float]:
        ts = self.loader.camera_timestamps
        assert ts, "Camera timestamps list is empty"
        return ts

    @cached_property
    def velo_times(self) -> List[float]:
        ts = self.loader.velo_timestamps
        assert ts, "LiDAR timestamps list is empty"
        return ts

    @cached_property
    def threshold(self) -> float:
        if self._user_threshold is not None:
            assert self._user_threshold > 0, "Threshold must be positive"
            return self._user_threshold
        dt_cam = np.diff(self.cam_times)
        dt_velo = np.diff(self.velo_times)
        med_cam = float(np.median(dt_cam))
        med_velo = float(np.median(dt_velo))
        assert med_cam > 0 and med_velo > 0, "Non-positive median delta"
        return DEFAULT_THRESHOLD_FACTOR * min(med_cam, med_velo)

    @cached_property
    def matcher(self) -> TimeMatcher:
        return TimeMatcher(self.threshold)

    def match_pairs(self) -> List[Tuple[int, int]]:
        return self.matcher.match_pairs(self.cam_times, self.velo_times)
