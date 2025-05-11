from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

class TimestampLoader:
    """
    Загружает и парсит файлы timestamps.txt для камеры и LiDAR.
    """

    def __init__(self, raw_root: Path, cam_folder: str):
        self.raw_root = Path(raw_root)
        self.cam_folder = cam_folder

    def _parse_timestamp(self, line: str) -> float:
        """
        Преобразует строку "YYYY-MM-DD hh:mm:ss.sss..." в секунды от начала суток.
        Бросает ValueError, если формат неверен.
        """
        line = line.strip()
        if not line:
            raise ValueError("Пустая строка таймстемпа")
        # отброс даты, если есть
        if ' ' in line:
            _, timestr = line.split(' ', 1)
        else:
            timestr = line
        parts = timestr.split(':')
        if len(parts) != 3:
            raise ValueError(f"Неправильный формат времени: {line}")
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)

    def load_camera_timestamps(self) -> List[float]:
        """
        Читает файл raw_root/cam_folder/timestamps.txt
        и возвращает список времён камер в секундах.
        """
        path = self.raw_root / self.cam_folder / 'timestamps.txt'
        if not path.exists():
            logger.error(f"Файл не найден: {path}")
            raise FileNotFoundError(f"{path} does not exist")
        with open(path, 'r') as f:
            lines = [ln for ln in f if ln.strip()]
        return [self._parse_timestamp(ln) for ln in lines]

    def load_velo_timestamps(self) -> List[float]:
        """
        Читает файл raw_root/velodyne_points/timestamps.txt
        и возвращает список времён сканов LiDAR в секундах.
        """
        path = self.raw_root / 'velodyne_points' / 'timestamps.txt'
        if not path.exists():
            logger.error(f"Файл не найден: {path}")
            raise FileNotFoundError(f"{path} does not exist")
        with open(path, 'r') as f:
            lines = [ln for ln in f if ln.strip()]
        return [self._parse_timestamp(ln) for ln in lines]
