# synchronizer/time_utils.py
from typing import Final
import logging

logger = logging.getLogger(__name__)

# Если где-то понадобится менять EPS для сравнения — здесь его и настраивать
EPS_SECONDS: Final[float] = 1e-6

def parse_timestamp(line: str) -> float:
    """
    Преобразует строку вида "YYYY-MM-DD hh:mm:ss.sss…" или "hh:mm:ss.sss…"
    в секунды с начала суток (float).

    :raises ValueError: если строка пустая или имеет некорректный формат.
    """
    s = line.strip()
    if not s:
        raise ValueError("Нельзя распарсить пустую строку таймстемпа")
    # отброс даты, если есть
    if ' ' in s:
        _, s = s.split(' ', 1)
    parts = s.split(':')
    if len(parts) != 3:
        raise ValueError(f"Неправильный формат времени: '{line}'")
    h, m, sec = parts
    try:
        hours = int(h)
        minutes = int(m)
        seconds = float(sec)
    except Exception as e:
        raise ValueError(f"Ошибка при разборе частей времени '{line}': {e}") from e
    total = hours * 3600 + minutes * 60 + seconds
    # Защита от нуля/отрицательных
    if total < EPS_SECONDS:
        logger.debug(f"parse_timestamp: значение времени слишком близко к нулю ({total}), "
                     f"используется EPS={EPS_SECONDS}")
        total = EPS_SECONDS
    return total
