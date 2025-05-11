# src/synchronizer/synchronizer.py

from pathlib import Path
from typing import List, Tuple, Optional
import bisect

import numpy as np


class Synchronizer:
    """
    Синхронизирует по времени кадры камеры, сканы лидара и замеры IMU из KITTI raw.
    Возвращает список кортежей (i_cam, i_velo, i_imu), отнумерованных от нуля.
    """

    def __init__(
        self,
        raw_root: Path,
        cam_folder: str,
        max_delta: Optional[float] = None,
    ):
        """
        :param raw_root:   корень папки с KITTI‐raw данными,
                          внутри должны лежать папки:
                            cam_folder/data,
                            velodyne_points/data + timestamps.txt,
                            oxts/timestamps.txt
        :param cam_folder: название каталога с камерами, например "image_02"
        :param max_delta:  если None — порог вычисляется автоматически как
                          0.5 * min(median_delta_cam, median_delta_velo)
        """
        self.root       = Path(raw_root)
        self.cam_folder = cam_folder

        # извлекаем индекс камеры из имени, eg. "image_02" → 2
        try:
            self.cam_idx = int(cam_folder.split('_')[-1])
        except ValueError:
            raise ValueError(f"Неверный cam_folder: {cam_folder}")

        # ленивые буферы
        self._cam_ts   = None  # type: Optional[List[float]]
        self._velo_ts  = None  # type: Optional[List[float]]
        self._imu_ts   = None  # type: Optional[List[float]]

        # порог на расхождение
        if max_delta is None:
            cam_ts  = self._load_camera_timestamps()
            velo_ts = self._load_velo_timestamps()
            dt_cam  = np.diff(cam_ts)
            dt_velo = np.diff(velo_ts)
            med_cam  = float(np.median(dt_cam))
            med_velo = float(np.median(dt_velo))
            self.threshold = 0.5 * min(med_cam, med_velo)
            print(f"[Synchronizer] auto-threshold={self.threshold:.6f}s "
                  f"(½·min(med_cam={med_cam:.3f}, med_velo={med_velo:.3f}))")
        else:
            self.threshold = float(max_delta)

    def _to_seconds(self, timestr: str) -> float:
        """
        "YYYY-MM-DD hh:mm:ss.sss..." или "hh:mm:ss.sss..." → секунды с начала суток.
        """
        if ' ' in timestr:
            _, timestr = timestr.split(' ', 1)
        h, m, s = timestr.split(':')
        return int(h) * 3600 + int(m) * 60 + float(s)

    def _load_camera_timestamps(self) -> List[float]:
        fn = self.root / self.cam_folder / 'timestamps.txt'
        with open(fn, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return [self._to_seconds(l) for l in lines]

    def _load_velo_timestamps(self) -> List[float]:
        start_fn = self.root / 'velodyne_points' / 'timestamps_start.txt'
        end_fn = self.root / 'velodyne_points' / 'timestamps_end.txt'
        with open(start_fn) as fs, open(end_fn) as fe:
            starts = [l.strip() for l in fs if l.strip()]
            ends = [l.strip() for l in fe if l.strip()]
        ts_start = [self._to_seconds(l) for l in starts]
        ts_end = [self._to_seconds(l) for l in ends]
        return [(s + e) / 2.0 for s, e in zip(ts_start, ts_end)]

    def _load_imu_timestamps(self) -> List[float]:
        fn = self.root / 'oxts' / 'timestamps.txt'
        if not fn.exists():
            raise FileNotFoundError(f"IMU timestamps not found: {fn}")
        with open(fn, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        return [self._to_seconds(l) for l in lines]

    def sync(self) -> List[Tuple[int, int, int]]:
        """
        Для каждого изображения находит ближайший скан лидара и запись IMU
        внутри self.threshold. Возвращает список (i_cam, i_velo, i_imu),
        все индексы сдвинуты так, что первая удачная тройка — (0,0,0).
        """
        if self._cam_ts   is None:
            self._cam_ts   = self._load_camera_timestamps()
        if self._velo_ts  is None:
            self._velo_ts  = self._load_velo_timestamps()
        if self._imu_ts   is None:
            self._imu_ts   = self._load_imu_timestamps()

        matches: List[Tuple[int,int,int]] = []

        for i, t_cam in enumerate(self._cam_ts):
            # найти ближайший лидарный скан
            j = bisect.bisect_left(self._velo_ts, t_cam)
            candidates = [c for c in (j - 1, j, j + 1) if 0 <= c < len(self._velo_ts)]
            best_velo = min(candidates, key=lambda c: abs(self._velo_ts[c] - t_cam), default=None)
            if best_velo is None or abs(self._velo_ts[best_velo] - t_cam) > self.threshold:
                continue

            # найти ближайшую IMU-запись
            k = bisect.bisect_left(self._imu_ts, t_cam)
            best_imu = None
            for cand in (k-1, k):
                if 0 <= cand < len(self._imu_ts):
                    if best_imu is None or abs(self._imu_ts[cand] - t_cam) < abs(self._imu_ts[best_imu] - t_cam):
                        best_imu = cand
            if best_imu is None or abs(self._imu_ts[best_imu] - t_cam) > self.threshold:
                continue

            matches.append((i, best_velo, best_imu))

        if not matches:
            return []

        # Сдвигаем индексы так, чтобы первая совпавшая тройка стала (0,0,0)
        i0, _, _ = matches[0]
        return [(i - i0, j, k) for (i, j, k) in matches]