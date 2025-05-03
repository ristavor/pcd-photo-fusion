import logging
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

def project_points(
    pts: np.ndarray, R: np.ndarray, T: np.ndarray, K: np.ndarray
) -> np.ndarray:
    """
    Проецирует LiDAR-точки pts (N×3) в пиксели изображения.
    Возвращает массив uvz (N×3): [u, v, z_cam].
    """
    P_cam = (R @ pts.T + T[:, None]).T      # N×3
    uvw   = (K @ P_cam.T).T                 # N×3

    # Защита от деления на ноль
    eps = 1e-6
    uvw[:, 2] = np.where(np.abs(uvw[:,2])<eps, eps, uvw[:,2])

    u = uvw[:, 0] / uvw[:, 2]
    v = uvw[:, 1] / uvw[:, 2]
    z = P_cam[:, 2]
    return np.vstack((u, v, z)).T           # N×3

def _bilinear_interpolate(
    img_float: np.ndarray,
    uv: np.ndarray
) -> np.ndarray:
    """
    Билинейная интерполяция цвета из img_float по вещественным uv (M×2).
    img_float: H×W×3 float32 [0..1].
    Возвращает RGB float32 [0..1].
    """
    h, w = img_float.shape[:2]
    u, v = uv[:,0], uv[:,1]

    x0 = np.floor(u).astype(int)
    y0 = np.floor(v).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1

    in_bounds = (x0 >= 0)&(x1 < w)&(y0 >= 0)&(y1 < h)
    oob_count = np.count_nonzero(~in_bounds)
    if oob_count:
        logger.warning(f"[colorize] {oob_count} точек вне изображения → дефолтный цвет")

    x0c = np.clip(x0, 0, w-1)
    x1c = np.clip(x1, 0, w-1)
    y0c = np.clip(y0, 0, h-1)
    y1c = np.clip(y1, 0, h-1)

    du = (u - x0).astype(np.float32)
    dv = (v - y0).astype(np.float32)

    wa = (1 - du)*(1 - dv)
    wb =    du *(1 - dv)
    wc = (1 - du)*    dv
    wd =    du *    dv

    Ia = img_float[y0c, x0c]
    Ib = img_float[y0c, x1c]
    Ic = img_float[y1c, x0c]
    Id = img_float[y1c, x1c]

    colors = (Ia*wa[:,None] + Ib*wb[:,None] + Ic*wc[:,None] + Id*wd[:,None]).astype(np.float32)
    colors = colors[:, ::-1]  # BGR→RGB

    if oob_count:
        colors[~in_bounds] = 0.0

    return colors

def colorize(
    xyz: np.ndarray,
    img: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    K: np.ndarray
) -> o3d.geometry.PointCloud:
    """
    Возвращает цветное PointCloud:
      1) проектируем → uvz
      2) фильтруем z>0 и внутри кадра
      3) нормализуем img → img_float
      4) билинейно интерполируем
      5) собираем o3d.geometry.PointCloud
    """
    uvz = project_points(xyz, R, T, K)

    h, w = img.shape[:2]
    u,v,z = uvz[:,0], uvz[:,1], uvz[:,2]
    mask = (z>0)&(u>=0)&(u<w)&(v>=0)&(v<h)
    uv_vis  = uvz[mask,:2]
    xyz_vis = xyz[mask]

    img_float = img.astype(np.float32) / 255.0

    colors = _bilinear_interpolate(img_float, uv_vis)

    pcd = o3d.geometry.PointCloud()
    # явный contiguous для Open3D
    pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz_vis))
    pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors))

    return pcd
