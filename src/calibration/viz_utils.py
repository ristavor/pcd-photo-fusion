import cv2
import numpy as np
from typing import Tuple
from numpy.typing import NDArray


def reproject_and_show(
    lidar_points: NDArray[np.float64],
    rvec: NDArray[np.float64],
    tvec: NDArray[np.float64],
    K: NDArray[np.float64],
    D: NDArray[np.float64],
    image: NDArray[np.uint8],
    window_name: str = "Overlay",
    point_color: Tuple[int, int, int] = (0, 255, 0),
    point_size: int = 2
) -> None:
    """
    Проецирует входные LiDAR-точки (N×3) на изображение image (BGR) с заданными
    rvec, tvec, intrinsics K и distCoeffs D. Рисует перекрашенные кружки поверх
    image и показывает результат в окне window_name, не возвращая кадр.

    Параметры:
      lidar_points (NDArray[float64], shape (N,3)): XYZ LiDAR-точки.
      rvec (NDArray[float64], shape (3,1) или (3,)): Rodrigues-вектор.
      tvec (NDArray[float64], shape (3,1) или (3,)): вектор трансляции.
      K (NDArray[float64], shape (3,3)): матрица внутренних параметров камеры.
      D (NDArray[float64], shape (5,)): коэффициенты дисторсии.
      image (NDArray[uint8], shape (H,W,3)): BGR-изображение.
      window_name (str): имя окна OpenCV для imshow.
      point_color (Tuple[int,int,int]): BGR-цвет кружков (по умолчанию зелёный).
      point_size (int): радиус кружков в пикселях.

    Примечание:
      — Функция вызывает cv2.imshow и сразу же cv2.waitKey(1), что обновляет окно.
      — Не сохраняет и не возвращает кадр; просто отображает.
    """
    # Убедимся, что входящие данные имеют правильный тип
    pts = lidar_points.reshape(-1, 3).astype(np.float64)

    # Проецируем точки (N×3) → (N×1×2) + глубину
    uvz, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    uv = uvz.reshape(-1, 2)
    z = uvz.reshape(-1, 3)[:, 2]

    h, w = image.shape[:2]
    vis = image.copy()

    # Рисуем только те точки, у которых z > 0 и (u,v) в пределах кадра
    for (u, v), depth in zip(uv, z):
        if depth <= 0:
            continue
        ui = int(round(u))
        vi = int(round(v))
        if 0 <= ui < w and 0 <= vi < h:
            cv2.circle(vis, (ui, vi), point_size, point_color, -1)

    cv2.imshow(window_name, vis)
    cv2.waitKey(1)


def make_overlay_image(
    all_lidar: NDArray[np.float64],
    rvec: NDArray[np.float64],
    tvec: NDArray[np.float64],
    K: NDArray[np.float64],
    D: NDArray[np.float64],
    image: NDArray[np.uint8],
    gamma: float = 0.3  # Параметр для усиления градиента
) -> NDArray[np.uint8]:
    """
    Возвращает NumPy-изображение overlay (BGR), в котором на исходное image
    наложены точки LiDAR по rvec/tvec. Цвет точек зависит от относительной глубины:
    зеленый - ближайшие точки в кадре, красный - самые дальние точки в кадре.

    Параметры:
      all_lidar (NDArray[float64], shape (N,3)): XYZ LiDAR-точки.
      rvec (NDArray[float64], shape (3,1) или (3,)): Rodrigues-вектор.
      tvec (NDArray[float64], shape (3,1) или (3,)): вектор трансляции.
      K (NDArray[float64], shape (3,3)): матрица внутренних параметров камеры.
      D (NDArray[float64], shape (5,)): коэффициенты дисторсии.
      image (NDArray[uint8], shape (H,W,3)): BGR-изображение.
      gamma (float): параметр для усиления градиента (0.1-1.0).
        Меньшие значения дают более резкий переход цветов.

    Возвращает:
      overlay (NDArray[uint8], shape (H,W,3)): кадр с точками, окрашенными
        в зависимости от относительной глубины (без блокирующего waitKey).
    """
    pts = all_lidar.reshape(-1, 3).astype(np.float64)

    proj_uv, _ = cv2.projectPoints(pts, rvec, tvec, K, D)
    uv = proj_uv.reshape(-1, 2)

    # Вычисляем глубину в камере: Z = (R * X + T)_z
    R_mat = cv2.Rodrigues(rvec)[0]  # (3×3)
    P_cam = (R_mat @ all_lidar.T + tvec).T  # (N×3)
    z = P_cam[:, 2]

    # Вычисляем углы поля зрения камеры
    fov_x = 2 * np.arctan2(K[0,2], K[0,0])  # горизонтальный угол обзора
    fov_y = 2 * np.arctan2(K[1,2], K[1,1])  # вертикальный угол обзора

    # Вычисляем углы для каждой точки относительно оптической оси камеры
    x_cam = P_cam[:, 0]
    y_cam = P_cam[:, 1]
    angles_x = np.arctan2(x_cam, z)
    angles_y = np.arctan2(y_cam, z)

    h, w = image.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]

    # Комбинируем все условия фильтрации:
    # 1. Положительная глубина
    # 2. Точка в пределах изображения
    # 3. Точка в пределах поля зрения камеры
    mask = (
        (z > 0) & 
        (u >= 0) & (u < w) & 
        (v >= 0) & (v < h) &
        (np.abs(angles_x) < fov_x/2) &
        (np.abs(angles_y) < fov_y/2)
    )

    u_vis = u[mask].astype(np.int32)
    v_vis = v[mask].astype(np.int32)
    z_vis = z[mask]

    # Если после фильтрации не осталось точек, возвращаем исходное изображение
    if len(z_vis) == 0:
        return image.copy()

    # Находим минимальную и максимальную глубину среди видимых точек
    min_z = np.min(z_vis)
    max_z = np.max(z_vis)
    
    # Нормализуем глубину относительно видимых точек (0 - близко, 1 - далеко)
    z_normalized = (z_vis - min_z) / (max_z - min_z)
    
    # Применяем степенное преобразование для усиления градиента
    z_normalized = np.power(z_normalized, gamma)
    
    # Создаем цвета: зеленый (близко) -> красный (далеко)
    # BGR формат: (0,255,0) - зеленый, (0,0,255) - красный
    colors = np.zeros((len(z_vis), 3), dtype=np.uint8)
    colors[:, 1] = (255 * (1 - z_normalized)).astype(np.uint8)  # G канал (зеленый)
    colors[:, 2] = (255 * z_normalized).astype(np.uint8)        # R канал (красный)

    overlay = image.copy()
    for (ui, vi), color in zip(zip(u_vis, v_vis), colors):
        overlay[vi, ui] = color

    return overlay


def draw_overlay(
    all_lidar: NDArray[np.float64],
    rvec: NDArray[np.float64],
    tvec: NDArray[np.float64],
    K: NDArray[np.float64],
    D: NDArray[np.float64],
    image: NDArray[np.uint8],
    window_name: str = "Overlay"
) -> None:
    """
    Быстрая отрисовка «облако → картинка» с наложением LiDAR-точек и текущей
    матрицы R (3×3) плюс вектора T.

    Порядок действий:
      1. Вызываем make_overlay_image, чтобы получить изображение с
         красными точками LiDAR.
      2. Рисуем поверх него три строки (R0, R1, R2) для матрицы вращения
         (каждая в формате "[r00, r01, r02]") чёрной обводкой и жёлтым фоном.
      3. Под ними рисуем строку T: "[tx, ty, tz]" с той же стилизацией.
      4. Вызываем cv2.imshow(window_name, overlay) и cv2.waitKey(1).

    Параметры:
      all_lidar (NDArray[float64], shape (N,3)): LiDAR-точки (XYZ).
      rvec (NDArray[float64], shape (3,1) или (3,)): Rodrigues-вектор.
      tvec (NDArray[float64], shape (3,1) или (3,)): вектор трансляции.
      K (NDArray[float64], shape (3,3)): intrinsics камеры.
      D (NDArray[float64], shape (5,)): distCoeffs камеры.
      image (NDArray[uint8], shape (H,W,3)): базовое изображение BGR.
      window_name (str): имя окна для отображения.

    Примечание:
      — Код внутри вычисляет R_mat = Rodrigues(rvec) дважды, но это незначительно
        с точки зрения скорости. Можно объединить, если нужно micro‐оптимиазация.
    """
    # Сначала получаем изображение с красными точками
    overlay = make_overlay_image(all_lidar, rvec, tvec, K, D, image)

    # Вычисляем матрицу R один раз
    R_mat = cv2.Rodrigues(rvec)[0]  # shape (3,3)
    tv = tvec.flatten()  # (3,)

    # Формируем текстовые строки для каждой строки матрицы R
    row0 = f"R0: [{R_mat[0,0]:.2f}, {R_mat[0,1]:.2f}, {R_mat[0,2]:.2f}]"
    row1 = f"R1: [{R_mat[1,0]:.2f}, {R_mat[1,1]:.2f}, {R_mat[1,2]:.2f}]"
    row2 = f"R2: [{R_mat[2,0]:.2f}, {R_mat[2,1]:.2f}, {R_mat[2,2]:.2f}]"
    t_line = f"T: [{tv[0]:.2f}, {tv[1]:.2f}, {tv[2]:.2f}]"

    # Настройки шрифта и стиля текста
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2

    # Рисуем R-строки (чёрная обводка, жёлтая заливка)
    for i, line in enumerate((row0, row1, row2)):
        y = 25 + 20 * i
        # Черная обводка
        cv2.putText(overlay, line, (10, y), font, font_scale,
                    (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
        # Желтая заливка (BGR = (0,255,255))
        cv2.putText(overlay, line, (10, y), font, font_scale,
                    (0, 255, 255), thickness, lineType=cv2.LINE_AA)

    # Рисуем строку T ниже строк R
    y_t = 25 + 20 * 3
    cv2.putText(overlay, t_line, (10, y_t), font, font_scale,
                (0, 0, 0), thickness + 2, lineType=cv2.LINE_AA)
    cv2.putText(overlay, t_line, (10, y_t), font, font_scale,
                (0, 255, 255), thickness, lineType=cv2.LINE_AA)

    # Отображаем результат
    cv2.imshow(window_name, overlay)
    cv2.waitKey(1)
