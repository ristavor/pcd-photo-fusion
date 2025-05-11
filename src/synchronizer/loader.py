from pathlib import Path
from typing import List
from .time_utils import parse_timestamp

class TimestampLoader:
    """
    Загружает и парсит файлы timestamps.txt для камеры и LiDAR.
    """

    def __init__(self, raw_root: Path, cam_folder: str):
        self.raw_root = Path(raw_root)
        self.cam_folder = cam_folder

    def load_camera_timestamps(self) -> List[float]:
        """
        Читает файл raw_root/cam_folder/timestamps.txt
        и возвращает список времён камер в секундах.
        """
        path = self.raw_root / self.cam_folder / 'timestamps.txt'
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        with open(path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [parse_timestamp(ln) for ln in lines]

    def load_velo_timestamps(self) -> List[float]:
        """
        Читает файл raw_root/velodyne_points/timestamps.txt
        и возвращает список времён сканов LiDAR в секундах.
        """
        path = self.raw_root / 'velodyne_points' / 'timestamps.txt'
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        with open(path, 'r') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return [parse_timestamp(ln) for ln in lines]
