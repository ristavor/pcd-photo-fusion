# src/utils/constants.py
from typing import Final

# На сколько процентов от медианного дельта-времени берём порог
DEFAULT_THRESHOLD_FACTOR: Final[float] = 0.5

# Смещения кандидатов при поиске ближайшего по времени
CANDIDATE_OFFSETS: Final[tuple[int, int, int]] = (-1, 0, 1)

# Минимальное положительное время (защита от деления на ноль)
EPS_SECONDS: Final[float] = 1e-6
