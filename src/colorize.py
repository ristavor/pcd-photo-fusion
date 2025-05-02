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
    По проекциям uv (M×2) и изображению img возвращает массив цветов (M×3) в формате [0,1].
    """
    # img имеет формат H×W×3, uv[:,0]=u, uv[:,1]=v
    colors = img[uv[:,1], uv[:,0]] / 255.0  # нормируем uint8→float
    # Open3D ждёт цвета в формате R,G,B
    # Если img в BGR (OpenCV), сначала перекладываем в RGB:
    colors = colors[:, ::-1]
    return colors
