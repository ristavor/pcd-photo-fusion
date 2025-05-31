# calibration/roi_selector/viz_utils.py

import cv2
import numpy as np
import open3d as o3d


def reproject_and_show(
    lidar_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    image: np.ndarray,
    window_name: str = "Overlay",
    point_color: tuple[int, int, int] = (0, 255, 0),
    point_size: int = 2
) -> None:
    """
    Проецирует входные LiDAR-точки (Nx3) на изображение image (BGR) с заданными
    rvec, tvec, intrinsic K, distCoeffs D. Рисует перекрашенные кружки поверх
    image и показывает результат в окне window_name.
    """
    pts = lidar_points.reshape(-1, 3).astype(np.float64)

    # Сначала проецируем каждую точку
    uvz, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    uv = uvz.reshape(-1, 2)
    z = uvz.reshape(-1, 3)[:, 2]  # глубина камеры

    h, w = image.shape[:2]
    vis = image.copy()

    # Рисуем только те точки, у которых z > 0 (перед камерой) и лежат в пределах изображения
    for (u, v), depth in zip(uv, z):
        if depth <= 0:
            continue
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(vis, (ui, vi), point_size, point_color, -1)

    cv2.imshow(window_name, vis)
    cv2.waitKey(1)
