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

def draw_overlay(
    all_lidar: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    image: np.ndarray,
    window_name: str = "Overlay"
):
    """
    Быстрая отрисовка «облако → картинка» для текущего rvec + tvec.
    all_lidar: (N×3) LiDAR-точки
    rvec, tvec: (3×1) Rodrigues-вектор и фиксированный t
    K, D      : intrinsics
    image     : BGR numpy array
    window_name: имя окна для imshow
    """
    # 1) Получаем proj_uv (N×1×2) + высчитываем глубину вручную:
    proj_uv, _ = cv2.projectPoints(
        objectPoints=all_lidar.reshape(-1, 1, 3),
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=K,
        distCoeffs=D
    )
    uv = proj_uv.reshape(-1, 2)  # (N×2)

    # 2) Вычисляем глубину Z = (R * X + T)_z
    R_mat = cv2.Rodrigues(rvec)[0]        # (3×3)
    P_cam = (R_mat @ all_lidar.T + tvec).T  # (N×3)
    z = P_cam[:, 2]                        # (N,)

    # 3) Фильтруем только те точки, для которых:
    #    z>0 (точка «перед» камерой) и (u,v) лежат внутри кадра.
    h, w = image.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]
    mask = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u_vis = u[mask].astype(np.int32)
    v_vis = v[mask].astype(np.int32)

    # 4) Накладываем: перекрашиваем пиксели (v_vis, u_vis) в красный
    overlay = image.copy()
    overlay[v_vis, u_vis] = (0, 0, 255)

    cv2.imshow(window_name, overlay)