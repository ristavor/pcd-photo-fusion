# src/colorize.py
import numpy as np

def project_points(pts: np.ndarray, R: np.ndarray, T: np.ndarray, K: np.ndarray):
    """
    Проецирует LiDAR-точки pts (N×3) в пиксели изображения.
    Возвращает массив uv (N×2) и глубины z (N).
    """
    # Переводим в систему камеры
    P_cam = (R @ pts.T + T[:, None]).T    # N×3
    # Проецируем через K
    uvw = (K @ P_cam.T).T                # N×3
    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    return np.vstack((u, v, P_cam[:, 2])).T  # N×3: [u, v, z]

def assign_colors(uv: np.ndarray, img: np.ndarray):
    """
    uv: (M,2) float — вещественные пиксельные координаты [u,v].
    img: H×W×3 uint8.
    Возвращает colors: (M,3) в [0,1], формат RGB, интерполированно билинейно.
    """
    h, w = img.shape[:2]
    u = uv[:, 0]
    v = uv[:, 1]

    # 1) Целые индексы четырёх соседних пикселей
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    # 2) Доли отступа от x0,y0
    du = u - x0  # [0…1)
    dv = v - y0  # [0…1)

    # 3) Веса для каждого угла
    wa = (1 - du) * (1 - dv)  # верх-лево: (x0,y0)
    wb = du       * (1 - dv)  # верх-право: (x1,y0)
    wc = (1 - du) * dv        # низ-лево:  (x0,y1)
    wd = du       * dv        # низ-право:  (x1,y1)

    # 4) Берём цвета из четырёх точек
    Ia = img[y0, x0]  # shape (M,3)
    Ib = img[y0, x1]
    Ic = img[y1, x0]
    Id = img[y1, x1]

    # 5) Составляем итоговый цвет
    # приводим все к float и нормируем [0,1]
    colors = (
        Ia * wa[:, None]
      + Ib * wb[:, None]
      + Ic * wc[:, None]
      + Id * wd[:, None]
    ) / 255.0

    # 6) Переворачиваем BGR→RGB (если используем OpenCV-формат)
    return colors[:, ::-1]
