# calibration/roi_selector/calib_io.py

import numpy as np
from pathlib import Path
from utils.calib import read_kitti_cam_calib, read_velo_to_cam


def load_camera_params(
    calib_cam_path: str,
    cam_idx: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Читает параметры камеры из calib_cam_to_cam.txt и
    параметры LiDAR→Cam из calib_velo_to_cam.txt.
    Возвращает:
      - K: (3×3) intrinsic
      - D: (5,)  distortion
      - R_gt: (3×3) LiDAR→Cam вращение (эталонное)
      - T_gt: (3,)  LiDAR→Cam трансляция (эталонная)
    """
    # Читаем intrinsics + D + R_rect + P_rect (R_rect и P_rect тут не нужны).
    K, D, _, _ = read_kitti_cam_calib(calib_cam_path, cam_idx)

    # Файл calib_velo_to_cam.txt лежит рядом с calib_cam_to_cam.txt
    velo_path = Path(calib_cam_path).with_name("calib_velo_to_cam.txt")
    R_gt, T_gt = read_velo_to_cam(str(velo_path))

    return K.astype(np.float64), D.astype(np.float64), R_gt, T_gt


def compute_axes_transform() -> np.ndarray:
    """
    Возвращает матрицу R_axes (3×3), которая переводит LiDAR-координаты
    (x→вперёд, y→влево, z→вверх) в «сырые» координаты камеры
    (x→вправо, y→вниз, z→вперёд), как задано в KITTI.
    """
    return np.array([[0, -1, 0],
                     [0,  0, -1],
                     [1,  0,  0]], dtype=np.float64)
