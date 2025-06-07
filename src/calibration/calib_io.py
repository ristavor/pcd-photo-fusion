# calibration/calib_io.py

import numpy as np
from typing import Tuple

__all__ = [
    "load_camera_params",
    "compute_axes_transform"
]

def load_camera_params(
    image_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает идеальные (эталонные) параметры камеры:
      - K: матрица внутренних параметров (фокус = max(width, height), центр = (cx, cy))
      - D: нулевой вектор коэффициентов дисторсии
    :param image_shape: кортеж (height, width) входного изображения
    :return: K (3×3), D (5,)
    """
    h, w = image_shape
    f = float(max(w, h))
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([
        [f,   0.0, cx],
        [0.0, f,   cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    D = np.zeros((5,), dtype=np.float64)
    return K, D

def compute_axes_transform() -> np.ndarray:
    """
    Возвращает матрицу R_axes (3×3), преобразующую координаты
    из LiDAR‐системы (x→вперёд, y→влево, z→вверх)
    в «сырые» координаты камеры KITTI (x→вправо, y→вниз, z→вперёд).
    """
    return np.array([
        [ 0, -1,  0],
        [ 0,  0, -1],
        [ 1,  0,  0]
    ], dtype=np.float64)
