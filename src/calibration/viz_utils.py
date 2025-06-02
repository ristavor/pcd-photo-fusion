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
    Быстрая отрисовка «облако → картинка» для текущего rvec + tvec, с текстом R/T.
    all_lidar: (N×3) LiDAR-точки
    rvec, tvec: (3×1) Rodrigues-вектор и вектор трансляции
    K, D      : intrinsics
    image     : BGR numpy array
    window_name: имя окна для imshow
    """
    # 1) Проецируем все точки:
    proj_uv, _ = cv2.projectPoints(
        objectPoints=all_lidar.reshape(-1, 1, 3),
        rvec=rvec,
        tvec=tvec,
        cameraMatrix=K,
        distCoeffs=D
    )
    uv = proj_uv.reshape(-1, 2)  # (N×2)

    # 2) Вычисляем глубину Z = (R * X + T)_z
    R_mat = cv2.Rodrigues(rvec)[0]         # (3×3)
    P_cam = (R_mat @ all_lidar.T + tvec).T  # (N×3)
    z = P_cam[:, 2]                         # (N,)

    # 3) Отбираем только «передние» точки в пределах кадра:
    h, w = image.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]
    mask = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    u_vis = u[mask].astype(np.int32)
    v_vis = v[mask].astype(np.int32)

    # 4) Копируем исходное изображение и закрашиваем проецируемые точки красным:
    overlay = image.copy()
    overlay[v_vis, u_vis] = (0, 0, 255)

    # 5) Формируем текст с текущими значениями rvec и tvec
    #    Берём rvec и tvec как одномерные массивы, чтобы вывести по 3 компоненты.
    R_mat = cv2.Rodrigues(rvec)[0]  # shape (3,3)
    # Формируем три строки для матрицы
    row0 = f"R0: [{R_mat[0, 0]:.2f}, {R_mat[0, 1]:.2f}, {R_mat[0, 2]:.2f}]"
    row1 = f"R1: [{R_mat[1, 0]:.2f}, {R_mat[1, 1]:.2f}, {R_mat[1, 2]:.2f}]"
    row2 = f"R2: [{R_mat[2, 0]:.2f}, {R_mat[2, 1]:.2f}, {R_mat[2, 2]:.2f}]"

    # 2) Сюда распаковываем tvec
    tv = tvec.flatten()
    t_line = f"T: [{tv[0]:.2f}, {tv[1]:.2f}, {tv[2]:.2f}]"

    # 3) Настройки шрифта и цвета
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # 4) Рисуем R-строки с чёрной обводкой и жёлтой заливкой
    for i, line in enumerate([row0, row1, row2]):
        y = 25 + 20 * i
        cv2.putText(overlay, line, (10, y), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
        cv2.putText(overlay, line, (10, y), font, font_scale, (0, 255, 255), thickness, lineType=cv2.LINE_AA)

    # 5) Текст T тоже с обводкой + заливкой
    y_t = 25 + 20 * 3  # сразу после трёх строк R, т.е. смещаем вниз ниже row2
    cv2.putText(overlay, t_line, (10, y_t), font, font_scale, (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    cv2.putText(overlay, t_line, (10, y_t), font, font_scale, (0, 255, 255), thickness, lineType=cv2.LINE_AA)

    # --- далее обычный вызов imshow ---
    cv2.imshow(window_name, overlay)
    cv2.waitKey(1)