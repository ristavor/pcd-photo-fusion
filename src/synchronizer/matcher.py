import bisect
import logging
from typing import List, Tuple, Optional

from utils.constants import CANDIDATE_OFFSETS

logger = logging.getLogger(__name__)


class TimeMatcher:
    """
    Для каждого cam_time ищет ближайший velo_time в пределах threshold.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def _get_candidate_indices(self,
                               cam_time: float,
                               velo_times: List[float]
                               ) -> List[int]:
        """
        1) Вычисляем позицию bisect
        2) Берём соседей по offsets (-1,0,1)
        Именно здесь вынесено получение окон поиска.
        """
        pos = bisect.bisect_left(velo_times, cam_time)
        return [
            i for i in (pos + d for d in CANDIDATE_OFFSETS)
            if 0 <= i < len(velo_times)
        ]

    def _select_best_candidate(self,
                               cam_time: float,
                               velo_times: List[float],
                               candidates: List[int]
                               ) -> Optional[int]:
        """
        Из списка кандидатов выбираем индекс с минимальным |Δt|.
        Вынесено, чтобы отделить логику выбора от логики проверки порога.
        """
        if not candidates:
            return None
        return min(candidates, key=lambda i: abs(velo_times[i] - cam_time))

    def _within_threshold(self,
                          cam_time: float,
                          velo_time: float
                          ) -> bool:
        """
        Проверяет, что |velo_time - cam_time| ≤ threshold.
        Отдельный метод для прозрачности и возможного переопределения.
        """
        return abs(velo_time - cam_time) <= self.threshold

    def find_best(self,
                  cam_time: float,
                  velo_times: List[float]
                  ) -> Optional[int]:
        """
        Фасад:
          1) Получаем кандидатов
          2) Выбираем лучший
          3) Проверяем порог
        """
        candidates = self._get_candidate_indices(cam_time, velo_times)
        best = self._select_best_candidate(cam_time, velo_times, candidates)
        if best is not None and self._within_threshold(cam_time, velo_times[best]):
            return best
        if best is not None:
            logger.debug(
                f"Best candidate {best} out of threshold: "
                f"|{velo_times[best] - cam_time:.6f}| > {self.threshold:.6f}"
            )
        return None

    def match_pairs(self,
                    cam_times: List[float],
                    velo_times: List[float]
                    ) -> List[Tuple[int, int]]:
        """
        Для каждого времени камеры возвращает пару (cam_idx, velo_idx),
        используя метод find_best.
        """
        matches: List[Tuple[int, int]] = []
        for cam_idx, t_cam in enumerate(cam_times):
            velo_idx = self.find_best(t_cam, velo_times)
            if velo_idx is not None:
                matches.append((cam_idx, velo_idx))
        return matches
