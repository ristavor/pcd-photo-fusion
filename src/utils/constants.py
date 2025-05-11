# src/utils/constants.py

from typing import Final

# На сколько процентов от медианного дельта-времени берём порог
DEFAULT_THRESHOLD_FACTOR: Final[float] = 0.5

# Смещения кандидатов при поиске ближайшего по времени
CANDIDATE_OFFSETS: Final[tuple[int, int, int]] = (-1, 0, 1)

# Минимальное положительное время (защита от деления на ноль в parse_timestamp)
EPS_SECONDS: Final[float] = 1e-6

# Малое число для защиты от деления на ноль в проекции точек (Colorizer)
EPS_DIV: Final[float] = 1e-6

# Смещения для билинейной интерполяции (dx, dy) — 4 соседних пикселя
BILINEAR_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (0, 0),  # Ia
    (1, 0),  # Ib
    (0, 1),  # Ic
    (1, 1),  # Id
)
