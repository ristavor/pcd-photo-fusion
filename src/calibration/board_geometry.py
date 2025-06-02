import numpy as np
import open3d as o3d
from typing import Tuple
from numpy.typing import NDArray


def compute_board_frame(
    board_cloud: o3d.geometry.PointCloud
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """
    Вычисляет локальную систему координат для плоскости шахматной доски,
    заданной облаком точек board_cloud.

    Алгоритм:
      1. Берём массив всех точек (N×3) из board_cloud.
      2. Вычисляем центр (origin) как среднее значение по каждой оси.
      3. Строим ковариационную матрицу относительно origin.
      4. Находим собственные значения и векторы (PCA):
         - По убыванию собственных значений получаем главную ось (x_axis)
           и вторую ось (y_axis) на плоскости шахматки.
      5. Нормализуем x_axis и y_axis.
      6. Вычисляем нормаль плоскости как cross(x_axis, y_axis) и нормируем.
      7. Гарантируем, что нормаль «смотрит» в сторону LiDAR (вдоль −origin),
         проверяя dot(normal, −origin) и при необходимости меняя знак.

    Параметры:
      board_cloud (o3d.geometry.PointCloud): подмножество точек, содержащих
        только поверхность шахматной доски.

    Возвращает:
      origin (NDArray[float32], shape (3,)): центр доски (mean XYZ).
      x_axis (NDArray[float32], shape (3,)): ось X в локальной системе (PCA).
      y_axis (NDArray[float32], shape (3,)): ось Y в локальной системе (PCA).
      normal (NDArray[float32], shape (3,)): нормаль к плоскости шахматки,
        направленная «к» LiDAR.

    Возвращаемые типы — всегда float32 для совместимости с Open3D и OpenCV.
    """
    points = np.asarray(board_cloud.points, dtype=np.float32)
    origin = points.mean(axis=0)

    # Ковариационная матрица точек относительно origin
    cov_matrix = np.cov((points - origin).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Сортируем собственные векторы по убыванию собственных значений
    sorted_indices = np.argsort(eigenvalues)[::-1]
    x_axis = eigenvectors[:, sorted_indices[0]]
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = eigenvectors[:, sorted_indices[1]]
    y_axis = y_axis / np.linalg.norm(y_axis)

    # Нормаль к плоскости
    normal = np.cross(x_axis, y_axis)
    normal = normal / np.linalg.norm(normal)

    # Если нормаль направлена «от» LiDAR, переворачиваем
    if np.dot(normal, -origin) < 0:
        normal = -normal

    return origin.astype(np.float32), x_axis.astype(np.float32), y_axis.astype(np.float32), normal.astype(np.float32)


def generate_object_points(
    origin: NDArray[np.float32],
    x_axis: NDArray[np.float32],
    y_axis: NDArray[np.float32],
    pattern_size: Tuple[int, int],
    square_size: float
) -> NDArray[np.float32]:
    """
    Генерирует 3D‐координаты внутренних углов шахматной доски
    в локальной системе координат доски.

    Каждый угол задаётся как:
        origin + i * square_size * x_axis + j * square_size * y_axis

    Параметры:
      origin (NDArray[float32], shape (3,)): центр доски.
      x_axis (NDArray[float32], shape (3,)): локальная ось X доски.
      y_axis (NDArray[float32], shape (3,)): локальная ось Y доски.
      pattern_size (Tuple[int, int]): (cols, rows) — число внутренних углов
        по X и Y (обычно, например, (7, 5)).
      square_size (float): размер одной клетки шахматки в метрах.

    Возвращает:
      pts3d (NDArray[float32], shape (cols*rows, 3)): массив XYZ‐координат
        всех углов в порядке по строкам (j от 0 до rows−1, i от 0 до cols−1).
    """
    cols, rows = pattern_size
    pts3d = np.zeros((cols * rows, 3), dtype=np.float32)
    idx = 0
    for j in range(rows):
        for i in range(cols):
            pts3d[idx] = origin + i * square_size * x_axis + j * square_size * y_axis
            idx += 1
    return pts3d


def refine_3d_corners(
    obj_pts: NDArray[np.float32],
    board_cloud: o3d.geometry.PointCloud,
    k: int = 20
) -> NDArray[np.float32]:
    """
    Для каждого сгенерированного 3D‐угла obj_pts находит k ближайших точек
    в облаке board_cloud, строит локальную плоскость (PCA) и проецирует
    исходный угол на эту плоскость. Возвращает уточнённые координаты углов.

    Логика:
      1. Строим KD‐дерево на board_cloud (только один раз).
      2. Для каждой точки p в obj_pts:
         a) Ищем индексы k ближайших точек в board_cloud.
         b) Берём их координаты (nei) и вычисляем центр этих точек.
         c) Строим ковариацию nei и находим собственный вектор,
            соответствующий наименьшему собственному значению (нормаль).
         d) Проецируем p на найденную плоскость: proj = p − dot(p − cen, normal) * normal.
      3. Собираем уточнённые точки proj в массив.

    Параметры:
      obj_pts (NDArray[float32], shape (M,3)): исходные 3D‐координаты углов шахматки.
      board_cloud (o3d.geometry.PointCloud): облако точек, содержащее только
        поверхности шахматной доски (ROI).
      k (int): число ближайших соседей для оценки локальной плоскости (по умолчанию 20).

    Возвращает:
      refined_pts (NDArray[float32], shape (M,3)): уточнённые координаты углов.
    """
    # Построение KD‐дерева один раз:
    pcd_points = np.asarray(board_cloud.points, dtype=np.float32)
    tree = o3d.geometry.KDTreeFlann(board_cloud)

    refined_pts = np.zeros_like(obj_pts, dtype=np.float32)
    for idx_pt, p in enumerate(obj_pts):
        # Находим k ближайших соседей p
        _, neighbor_indices, _ = tree.search_knn_vector_3d(p.astype(np.float64), k)
        nei = pcd_points[neighbor_indices]

        # Центр ближайших точек
        centroid = nei.mean(axis=0)
        cov_nei = np.cov((nei - centroid).T)
        eigvals, eigvecs = np.linalg.eigh(cov_nei)

        # Нормаль – собственный вектор с наименьшим собственным значением
        normal = eigvecs[:, np.argmin(eigvals)]
        normal = normal / np.linalg.norm(normal)

        # Проецируем p на локальную плоскость
        proj = p - np.dot(p - centroid, normal) * normal
        refined_pts[idx_pt] = proj.astype(np.float32)

    return refined_pts
