# src/rectifier/calib.py
import numpy as np
from pathlib import Path

def read_kitti_cam_calib(calib_path: str, cam_idx: int):
    """
    calib_path: путь до calib_cam_to_cam.txt
    cam_idx:    0..3 — номер камеры в KITTI
    Возвращает:
      K      — (3×3) intrinsic matrix,
      D      — (5,)   distortion coefficients,
      R_rect — (3×3) rectification rotation,
      P_rect — (3×4) projection matrix.
    """
    data = {}
    with open(calib_path, 'r') as f:
        for line in f:
            key, vals = line.split(':',1)
            data[key.strip()] = np.fromstring(vals, sep=' ')
    # строковый суффикс с двумя цифрами, например "02"
    idx = f'{cam_idx:02d}'

    # Intrinsic
    K = data[f'K_{idx}'].reshape(3,3)
    # Distortion (k1,k2,p1,p2,k3)
    D = data[f'D_{idx}']
    # Rectification rotation
    R_rect = data[f'R_rect_{idx}'].reshape(3,3)
    # Projection matrix
    P_rect = data[f'P_rect_{idx}'].reshape(3,4)

    return K, D, R_rect, P_rect
