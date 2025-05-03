import numpy as np
import open3d as o3d

def project_points(pts: np.ndarray, R: np.ndarray, T: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Проецирует LiDAR-точки pts (N×3) в пиксели изображения.
    Возвращает массив uvz (N×3): [u, v, z_cam].
    """
    P_cam = (R @ pts.T + T[:, None]).T    # N×3
    uvw   = (K @ P_cam.T).T               # N×3
    u     = uvw[:, 0] / uvw[:, 2]
    v     = uvw[:, 1] / uvw[:, 2]
    z     = P_cam[:, 2]
    return np.vstack((u, v, z)).T         # N×3

def _bilinear_interpolate(img: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """
    Билинейная интерполяция цвета в img по вещественным координатам uv (M×2).
    Возвращает цвета (M×3) в формате RGB, float в [0,1].
    """
    h, w = img.shape[:2]
    u, v = uv[:,0], uv[:,1]

    # Целые индексы четырёх соседних пикселей
    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    # Маска точек внутри изображения
    in_bounds = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
    # Логирование «выпавших»
    num_oob = np.count_nonzero(~in_bounds)
    if num_oob:
        print(f"[colorize] {num_oob} точек вне изображения → дефолтный цвет")

    # Чтобы не выйти за границы, зажимаем индексы
    x0c = np.clip(x0, 0, w-1)
    x1c = np.clip(x1, 0, w-1)
    y0c = np.clip(y0, 0, h-1)
    y1c = np.clip(y1, 0, h-1)

    # Доли смещения
    du = u - x0
    dv = v - y0

    # Веса
    wa = (1 - du) * (1 - dv)
    wb =    du   * (1 - dv)
    wc = (1 - du) *    dv
    wd =    du   *    dv

    # Вынимаем цвета (uint8 BGR → float)
    Ia = img[y0c, x0c].astype(float)
    Ib = img[y0c, x1c].astype(float)
    Ic = img[y1c, x0c].astype(float)
    Id = img[y1c, x1c].astype(float)

    # Билинейная интерполяция
    colors = (
          Ia * wa[:,None]
        + Ib * wb[:,None]
        + Ic * wc[:,None]
        + Id * wd[:,None]
    ) / 255.0  # в [0,1]

    # BGR → RGB
    colors = colors[:, ::-1]

    # Присваиваем дефолтный цвет «выпавшим»
    if num_oob:
        default_rgb = np.array([0,0,0], dtype=float)  # чёрный
        colors[~in_bounds] = default_rgb

    return colors


def colorize(
    xyz: np.ndarray,
    img: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    K: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    1) Проецирует xyz → uvz
    2) Оставляет лишь z>0 и в кадре
    3) Билинейно окрашивает видимые точки
    4) Возвращает готовый o3d.geometry.PointCloud
    """
    # 1) Проекция
    uvz = project_points(xyz, R, T, K)   # N×3

    # 2) Фильтрация по кадру и z>0
    h, w      = img.shape[:2]
    u, v, z   = uvz[:,0], uvz[:,1], uvz[:,2]
    mask      = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv_vis    = uvz[mask,:2]             # M×2
    xyz_vis   = xyz[mask]                # M×3

    # 3) Билинейная интерполяция цветов
    colors    = _bilinear_interpolate(img, uv_vis)  # M×3 float [0..1]

    # 4) Собираем PCD
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_vis)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd
