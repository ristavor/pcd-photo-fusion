from pathlib import Path
from typing import List
from .time_utils import parse_timestamp
from functools import cached_property

class TimestampLoader:
    """
    Загружает и парсит файлы timestamps.txt для камеры и LiDAR.
    """

    def __init__(self, raw_root: Path, cam_folder: str):
        self.raw_root = Path(raw_root)
        self.cam_folder = cam_folder

    @cached_property
    def camera_timestamps(self) -> List[float]:
        """
        Загружает и кэширует список времён камер.
        """
        path = self.raw_root / self.cam_folder / 'timestamps.txt'
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        with open(path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [parse_timestamp(ln) for ln in lines]

    @cached_property
    def velo_timestamps(self) -> List[float]:
        """
        Загружает и кэширует список времён LiDAR.
        """
        path = self.raw_root / 'velodyne_points' / 'timestamps.txt'
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        with open(path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [parse_timestamp(ln) for ln in lines]
