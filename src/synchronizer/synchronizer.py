# sync.py
from pathlib import Path
from typing import List, Tuple
import bisect
import numpy as np

class Synchronizer:
    """
    Синхронизирует timestamps камер и LiDAR-облаков по их файлам timestamps.txt в raw-данных.
    Возвращает список (cam_idx, velo_idx) для всех пар, где |t_cam - t_velo| ≤ threshold.
    """
    def __init__(self, raw_root: Path, cam_folder: str, threshold: float = None):
        self.raw_root   = Path(raw_root)
        self.cam_folder = cam_folder

        self.cam_ts  = self._load_camera_timestamps()
        self.velo_ts = self._load_velo_timestamps()

        if threshold is None:
            dt_cam   = np.diff(self.cam_ts)
            dt_velo  = np.diff(self.velo_ts)
            self.threshold = 0.5 * min(np.median(dt_cam), np.median(dt_velo))
        else:
            self.threshold = threshold

    def _to_seconds(self, timestr: str) -> float:
        if ' ' in timestr:
            _, timestr = timestr.split(' ', 1)
        h, m, s = timestr.split(':')
        return int(h)*3600 + int(m)*60 + float(s)

    def _load_camera_timestamps(self) -> List[float]:
        fn = self.raw_root / self.cam_folder / 'timestamps.txt'
        with open(fn, 'r') as f:
            return [self._to_seconds(l.strip()) for l in f if l.strip()]

    def _load_velo_timestamps(self) -> List[float]:
        fn = self.raw_root / 'velodyne_points' / 'timestamps.txt'
        with open(fn, 'r') as f:
            return [self._to_seconds(l.strip()) for l in f if l.strip()]

    def sync(self) -> List[Tuple[int,int]]:
        matches: List[Tuple[int,int]] = []
        for cam_i, t_cam in enumerate(self.cam_ts):
            j = bisect.bisect_left(self.velo_ts, t_cam)
            cands = [c for c in (j-1, j, j+1) if 0 <= c < len(self.velo_ts)]
            if not cands:
                continue
            best = min(cands, key=lambda c: abs(self.velo_ts[c] - t_cam))
            if abs(self.velo_ts[best] - t_cam) <= self.threshold:
                matches.append((cam_i, best))
        return matches
