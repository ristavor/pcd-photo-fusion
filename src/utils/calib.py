# src/utils/calib.py

from functools import lru_cache
from typing import Dict, Tuple
import json

import numpy as np


@lru_cache(maxsize=None)
def read_cam_to_cam(path: str) -> Dict[str, np.ndarray]:
    """
    Считывает calib_cam_to_cam.txt в словарь:
      ключи — части до ":",
      значения — np.array из чисел.
    Кешируется, чтобы файл читался лишь однажды.
    """
    data: Dict[str, np.ndarray] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            data[key.strip()] = np.fromstring(vals, sep=' ')
    return data


def read_velo_to_cam(path: str, is_kitti: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Считывает параметры R (3×3) и T (3,) из файла velo_to_cam.
    
    Параметры:
        path (str): путь к файлу калибровки
        is_kitti (bool): True для формата KITTI, False для JSON
        
    Возвращает:
        Tuple[np.ndarray, np.ndarray]: (R, T) где R - матрица 3x3, T - вектор 3x1
    """
    if is_kitti:
        # KITTI
        with open(path, 'r') as f:
            lines = f.readlines()
        data = {}
        for line in lines:
            if ':' in line:
                key, vals = line.split(':', 1)
                data[key.strip()] = np.fromstring(vals, sep=' ')
        R = data.get('R')
        T = data.get('T')
        if R is None or T is None:
            raise ValueError("Файл KITTI velo_to_cam должен содержать R и T")
        R = R.reshape(3, 3)
        T = T.reshape(3)
        return R, T
    else:
        with open(path, 'r') as f:
            data = json.load(f)
        # Поддержка альтернативных ключей
        R = data.get('R') or data.get('R_matrix')
        T = data.get('T') or data.get('T_vector')
        if R is None or T is None:
            raise ValueError("JSON must contain 'R' (или 'R_matrix') и 'T' (или 'T_vector')")
        R = np.array(R)
        T = np.array(T)
        return R, T


def read_kitti_cam_calib(path: str, cam_idx: int
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Берёт из словаря calib_cam_to_cam.txt параметры камеры #cam_idx:
      K       — (3×3) intrinsic,
      D       — (5,)  distortion,
      R_rect  — (3×3) rectification rotation,
      P_rect  — (3×4) projection matrix.
    """
    d = read_cam_to_cam(path)
    idx = f"{cam_idx:02d}"
    try:
        K = d[f"K_{idx}"].reshape(3, 3)
        D = d[f"D_{idx}"]
        R_rect = d[f"R_rect_{idx}"].reshape(3, 3)
        P_rect = d[f"P_rect_{idx}"].reshape(3, 4)
    except KeyError as e:
        raise KeyError(f"Ключ {e.args[0]} не найден в {path}") from None
    return K, D, R_rect, P_rect


def get_full_image_size(path: str, cam_idx: int) -> Tuple[int, int]:
    """
    Возвращает (width, height) = S_0X из calib_cam_to_cam.txt
    """
    d = read_cam_to_cam(path)
    idx = f"{cam_idx:02d}"
    arr = d.get(f"S_{idx}")
    if arr is None or arr.size < 2:
        raise KeyError(f"S_{idx} отсутствует в {path}")
    return int(arr[0]), int(arr[1])


def get_rectified_size(path: str, cam_idx: int) -> Tuple[int, int]:
    """
    Возвращает (width, height) = S_rect_0X из calib_cam_to_cam.txt
    """
    d = read_cam_to_cam(path)
    idx = f"{cam_idx:02d}"
    arr = d.get(f"S_rect_{idx}")
    if arr is None or arr.size < 2:
        raise KeyError(f"S_rect_{idx} отсутствует в {path}")
    return int(arr[0]), int(arr[1])
