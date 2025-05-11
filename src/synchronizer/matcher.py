# src/synchronizer/matcher.py
from typing import List, Tuple, Optional
import bisect
import logging

import numpy as np
from utils.constants import CANDIDATE_OFFSETS

logger = logging.getLogger(__name__)

class TimeMatcher:
    """
    Для каждого времени из cam_times ищет ближайшее в velo_times.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def _find_best(self, cam_time: float, velo_times: List[float]) -> Optional[int]:
        pos = bisect.bisect_left(velo_times, cam_time)
        candidates = [i for i in (pos + d for d in CANDIDATE_OFFSETS)
                      if 0 <= i < len(velo_times)]
        if not candidates:
            return None
        best = min(candidates, key=lambda i: abs(velo_times[i] - cam_time))
        if abs(velo_times[best] - cam_time) <= self.threshold:
            return best
        logger.debug(f"match 실패: |{velo_times[best]:.6f} - {cam_time:.6f}| > {self.threshold:.6f}")
        return None

    def match_pairs(self, cam_times: List[float], velo_times: List[float]) -> List[Tuple[int, int]]:
        matches: List[Tuple[int, int]] = []
        for i, t_cam in enumerate(cam_times):
            j = self._find_best(t_cam, velo_times)
            if j is not None:
                matches.append((i, j))
        return matches
