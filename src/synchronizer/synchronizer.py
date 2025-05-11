from pathlib import Path
from typing import List, Tuple
import numpy as np
from functools import cached_property

from .loader import TimestampLoader
from .matcher import TimeMatcher

class Synchronizer:
    """
    Фасад: загружает временные метки, вычисляет threshold и
    возвращает список синхронизованных пар (cam_idx, velo_idx).
    """

    def __init__(self, raw_root: Path, cam_folder: str,
                 threshold: float = None):
        """
        :param raw_root:   корень каталога raw-данных
        :param cam_folder: подпапка с изображениями, например "image_02"
        :param threshold:  порог совпадения в секундах (если None — вычисляется автоматически)
        """
        self.raw_root = Path(raw_root)
        self.cam_folder = cam_folder
        self._user_threshold = threshold
        # отложенная инициализация loader и matcher
        self._matcher = None

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
        """
        Ленивая инициализация порога: 0.5 * min(median Δcam, median Δvelo).
        Можно переопределить через аргумент __init__.
        """
        if self._user_threshold is not None:
            return self._user_threshold
        dt_cam = np.diff(self.cam_times)
        dt_velo = np.diff(self.velo_times)
        return 0.5 * min(float(np.median(dt_cam)), float(np.median(dt_velo)))

    @cached_property
    def matcher(self) -> TimeMatcher:
        return TimeMatcher(self.threshold)

    def match_pairs(self) -> List[Tuple[int, int]]:
        """
        Синхронизирует cam_times и velo_times, возвращая список (cam_idx, velo_idx).
        """
        return self.matcher.match_pairs(self.cam_times, self.velo_times)

