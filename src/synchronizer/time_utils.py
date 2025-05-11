# src/synchronizer/time_utils.py
import logging

from utils.constants import EPS_SECONDS

logger = logging.getLogger(__name__)


def parse_timestamp(line: str) -> float:
    """
    Преобразует "YYYY-MM-DD hh:mm:ss.sss…" или "hh:mm:ss.sss…" в секунды от начала суток.
    """
    s = line.strip()
    if not s:
        raise ValueError("Пустая строка таймстемпа")
    if ' ' in s:
        _, s = s.split(' ', 1)
    parts = s.split(':')
    if len(parts) != 3:
        raise ValueError(f"Неправильный формат времени: '{line}'")
    h, m, sec = parts
    try:
        total = int(h) * 3600 + int(m) * 60 + float(sec)
    except Exception as e:
        raise ValueError(f"Ошибка парсинга '{line}': {e}") from e
    # Защита от нуля/отрицательных
    if total < EPS_SECONDS:
        logger.debug(f"time_utils: значение {total:.6f}s меньше EPS, ставим EPS={EPS_SECONDS}")
        total = EPS_SECONDS
    return total
