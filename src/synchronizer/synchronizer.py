from pathlib import Path
from typing import List, Tuple
import bisect

import numpy as np


class Synchronizer:
    def __init__(
        self,
        raw_root: Path,
        cam_folder: str,
        max_delta: float | None = None,  # теперь None по умолчанию
    ):
        self.root       = Path(raw_root)
        self.cam_folder = cam_folder

        # извлечём индекс камеры из имени папки
        try:
            self.cam_idx = int(cam_folder.split('_')[-1])
        except:
            raise ValueError(f"Неверный cam_folder: {cam_folder}")

        # ленивые буферы
        self._cam_ts   = None
        self._velo_ts  = None
        self._imu_ts   = None

        # если порог не задан, посчитаем его автоматически
        if max_delta is None:
            # сразу загрузим камеру и лидар (для IMU не нужно)
            cam_ts  = self._load_camera_timestamps()
            velo_ts = self._load_velo_timestamps()
            # считаем дельты
            dt_cam  = np.diff(cam_ts)
            dt_velo = np.diff(velo_ts)
            # медиана
            med_cam  = float(np.median(dt_cam))
            med_velo = float(np.median(dt_velo))
            # порог = половина минимальной медианы
            self.threshold = 0.5 * min(med_cam, med_velo)
            print(f"[Synchronizer] auto-threshold={self.threshold:.6f}s "
                  f"(½·min(med_cam={med_cam:.3f}, med_velo={med_velo:.3f}))")
        else:
            self.threshold = max_delta


    def _to_seconds(self, timestr: str) -> float:
        """
        Преобразует "YYYY-MM-DD hh:mm:ss.sss..." или "hh:mm:ss.sss..." в секунды от начала суток.
        """
        if ' ' in timestr:
            _, timestr = timestr.split(' ', 1)
        h, m, s = timestr.split(':')
        return int(h)*3600 + int(m)*60 + float(s)

    def _load_camera_timestamps(self) -> List[float]:
        fn = self.root / self.cam_folder / 'timestamps.txt'
        with open(fn, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return [self._to_seconds(l) for l in lines]

    def _load_velo_timestamps(self) -> List[float]:
        fn = self.root / 'velodyne_points' / 'timestamps_start.txt'
        with open(fn, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return [self._to_seconds(l) for l in lines]

    def _load_imu_timestamps(self) -> List[float]:
        """
        Читает файл oxts/timestamps.txt → возвращает список секунд.
        """
        fn = self.root / 'oxts' / 'timestamps.txt'  # <-- поправили путь
        if not fn.exists():
            raise FileNotFoundError(f"IMU timestamps not found: {fn}")
        with open(fn, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return [self._to_seconds(l) for l in lines]

    def sync(self) -> List[Tuple[int,int,int]]:
        """
        Возвращает список кортежей (i_cam, i_velo, i_imu),
        где каждый элемент попал в диапазон self.threshold.
        """
        if self._cam_ts  is None: self._cam_ts  = self._load_camera_timestamps()
        if self._velo_ts is None: self._velo_ts = self._load_velo_timestamps()
        if self._imu_ts  is None: self._imu_ts  = self._load_imu_timestamps()

        matches = []
        for i, t_cam in enumerate(self._cam_ts):
            # лидар
            j = bisect.bisect_left(self._velo_ts, t_cam)
            best_velo = None
            for cand in (j-1, j):
                if 0 <= cand < len(self._velo_ts):
                    if best_velo is None or abs(self._velo_ts[cand] - t_cam) < abs(self._velo_ts[best_velo] - t_cam):
                        best_velo = cand
            if abs(self._velo_ts[best_velo] - t_cam) > self.threshold:
                continue

            # IMU
            k = bisect.bisect_left(self._imu_ts, t_cam)
            best_imu = None
            for cand in (k-1, k):
                if 0 <= cand < len(self._imu_ts):
                    if best_imu is None or abs(self._imu_ts[cand] - t_cam) < abs(self._imu_ts[best_imu] - t_cam):
                        best_imu = cand
            if abs(self._imu_ts[best_imu] - t_cam) > self.threshold:
                continue

            matches.append((i, best_velo, best_imu))

        return matches
