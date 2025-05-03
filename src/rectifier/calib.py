from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def read_cam_to_cam(path: str) -> dict[str, np.ndarray]:
    """
    Считывает calib_cam_to_cam.txt в словарь:
      ключи — строки до «:»,
      значения — NumPy-массивы.
    Результат кешируется, чтобы не читать файл при каждом вызове.
    """
    data: dict[str, np.ndarray] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            key, vals = line.split(':', 1)
            data[key.strip()] = np.fromstring(vals, sep=' ')
    return data


def read_kitti_cam_calib(path: str, cam_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Берёт из словаря параметры камеры #cam_idx:
      K      — (3×3) intrinsic matrix,
      D      — (5,)   distortion coefficients,
      R_rect — (3×3) rectification rotation,
      P_rect — (3×4) projection matrix.
    """
    d = read_cam_to_cam(path)
    idx = f'{cam_idx:02d}'
    try:
        K = d[f'K_{idx}'].reshape(3, 3)
        D = d[f'D_{idx}']  # (5,)
        R_rect = d[f'R_rect_{idx}'].reshape(3, 3)
        P_rect = d[f'P_rect_{idx}'].reshape(3, 4)
    except KeyError as e:
        raise KeyError(f"Ключ {e.args[0]} не найден в {path}") from None
    return K, D, R_rect, P_rect


def get_full_image_size(path: str, cam_idx: int) -> tuple[int, int]:
    """
    Возвращает (width, height) как S_0X из calib_cam_to_cam.txt
    """
    d = read_cam_to_cam(path)
    idx = f'{cam_idx:02d}'
    arr = d.get(f'S_{idx}')
    if arr is None or arr.size < 2:
        raise KeyError(f"S_{idx} отсутствует в {path}")
    return int(arr[0]), int(arr[1])


def get_rectified_size(path: str, cam_idx: int) -> tuple[int, int]:
    """
    Возвращает (width, height) как S_rect_0X из calib_cam_to_cam.txt
    """
    d = read_cam_to_cam(path)
    idx = f'{cam_idx:02d}'
    arr = d.get(f'S_rect_{idx}')
    if arr is None or arr.size < 2:
        raise KeyError(f"S_rect_{idx} отсутствует в {path}")
    return int(arr[0]), int(arr[1])
