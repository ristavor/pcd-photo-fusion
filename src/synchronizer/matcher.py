from typing import List, Tuple, Optional
import bisect
import numpy as np
import logging

logger = logging.getLogger(__name__)

CANDIDATE_OFFSETS = (-1, 0, +1)

class TimeMatcher:
    """
    Находит для каждого времени камеры ближайший LiDAR-скан в пределах threshold.
    """

    def __init__(self, threshold: float):
        """
        :param threshold: максимальное отклонение (в секундах) для совпадения.
        """
        self.threshold = threshold

    def _find_best(self,
                   cam_time: float,
                   velo_times: List[float]
                  ) -> Optional[int]:
        """
        Находит индекс самого близкого элемента в velo_times к cam_time.
        Возвращает None, если ближайшее расстояние > threshold.
        """
        pos = bisect.bisect_left(velo_times, cam_time)
        candidates = [i for i in (pos + d for d in CANDIDATE_OFFSETS)
                      if 0 <= i < len(velo_times)]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(velo_times[i] - cam_time))
        if abs(velo_times[best] - cam_time) <= self.threshold:
            return best
        logger.debug(f"Скан {best} отстоит на {abs(velo_times[best]-cam_time):.6f}s > threshold")
        return None

    def match_pairs(self,
                    cam_times: List[float],
                    velo_times: List[float]
                   ) -> List[Tuple[int, int]]:
        """
        Для каждого cam_times[i] возвращает пару (i, j), где j — ближайший scan.
        """
        matches: List[Tuple[int, int]] = []
        for i, t_cam in enumerate(cam_times):
            j = self._find_best(t_cam, velo_times)
            if j is not None:
                matches.append((i, j))
        return matches
