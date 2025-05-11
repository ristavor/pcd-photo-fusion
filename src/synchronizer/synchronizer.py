from pathlib import Path
from typing import List, Tuple
import numpy as np

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

        loader = TimestampLoader(self.raw_root, cam_folder)
        self.cam_times = loader.load_camera_timestamps()
        self.velo_times = loader.load_velo_timestamps()

        if threshold is None:
            dt_cam = np.diff(self.cam_times)
            dt_velo = np.diff(self.velo_times)
            threshold = 0.5 * min(float(np.median(dt_cam)), float(np.median(dt_velo)))
        self.matcher = TimeMatcher(threshold)

    def match_pairs(self) -> List[Tuple[int, int]]:
        """
        Выполняет синхронизацию и отбрасывает «плохие» кадры.
        """
        return self.matcher.match_pairs(self.cam_times, self.velo_times)
