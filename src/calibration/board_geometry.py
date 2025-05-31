# calibration/roi_selector/board_geometry.py

import numpy as np
import open3d as o3d


def compute_board_frame(
    board_cloud: o3d.geometry.PointCloud
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет локальную систему координат плоскости шахматки по
    облаку точек board_cloud, содержащее только точки доски.
    Возвращает:
      - origin: np.array(3,) = средняя точка
      - x_axis: np.array(3,) = главная ось (PCA)
      - y_axis: np.array(3,) = вторая ось (PCA)
      - normal: np.array(3,) = нормаль (x × y)
    """
    pts = np.asarray(board_cloud.points)
    origin = pts.mean(axis=0)

    # ковариация точек относительно origin
    cov = np.cov((pts - origin).T)
    vals, vecs = np.linalg.eigh(cov)
    # Сортируем собственные векторы по убыванию собственных значений
    idx = np.argsort(vals)[::-1]
    x_axis = vecs[:, idx[0]]
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = vecs[:, idx[1]]
    y_axis = y_axis / np.linalg.norm(y_axis)

    # нормаль как векторное произведение (x × y)
    normal = np.cross(x_axis, y_axis)
    normal = normal / np.linalg.norm(normal)

    # Чтобы нормаль «смотрела» в сторону LiDAR (вдоль -origin),
    # проверим знак: если dot(normal, -origin) < 0, то перевернём нормаль
    if np.dot(normal, -origin) < 0:
        normal = -normal

    return origin, x_axis, y_axis, normal


def generate_object_points(
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    pattern_size: tuple[int, int],
    square_size: float
) -> np.ndarray:
    """
    Генерирует 3D-корды углов шахматки в локальной СК доски:
      origin + i*square_size*x_axis + j*square_size*y_axis
    pattern_size=(cols, rows). Возвращает массив shape=(cols*rows, 3).
    """
    cols, rows = pattern_size
    pts3d = []
    for j in range(rows):
        for i in range(cols):
            pt = origin + i * square_size * x_axis + j * square_size * y_axis
            pts3d.append(pt)
    return np.array(pts3d, dtype=np.float32)


def refine_3d_corners(
    obj_pts: np.ndarray,
    board_cloud: o3d.geometry.PointCloud,
    k: int = 20
) -> np.ndarray:
    """
    Для каждой генерируемой 3D-точки (угла шахматки) ищет k ближайших точек
    в облаке board_cloud, оценивает локальную плоскость через PCA и проецирует
    исходную точку на эту плоскость. Возвращает уточнённый набор точек.
    """
    tree = o3d.geometry.KDTreeFlann(board_cloud)
    all_pts = np.asarray(board_cloud.points)
    refined = []

    for p in obj_pts:
        _, idx, _ = tree.search_knn_vector_3d(p, k)
        nei = all_pts[idx]
        cen = nei.mean(axis=0)
        cov = np.cov((nei - cen).T)
        vals, vecs = np.linalg.eigh(cov)
        # нормаль соответствует наименьшему собственному числу
        normal = vecs[:, np.argmin(vals)]
        proj = p - np.dot(p - cen, normal) * normal
        refined.append(proj)

    return np.array(refined, dtype=np.float32)
