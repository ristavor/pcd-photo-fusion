import logging
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)

class Colorizer:
    """
    Класс-обёртка над функцией colorize:
      1) Проекция
      2) Маскирование
      3) Нормализация изображения
      4) Билинейная интерполяция
      5) Сборка Open3D PointCloud
    """

    def __init__(self, R: np.ndarray, T: np.ndarray, K: np.ndarray):
        self.R = R
        self.T = T
        self.K = K

    def project_points(self, pts: np.ndarray) -> np.ndarray:
        """
        pts: (N×3) LiDAR-поинты.
        Возвращает uvz: (N×3) — [u, v, z_cam].
        """
        P_cam = (self.R @ pts.T + self.T[:, None]).T    # N×3
        uvw   = (self.K @ P_cam.T).T                   # N×3

        # Защита от деления на ноль:
        eps = 1e-6
        uvw[:, 2] = np.where(np.abs(uvw[:, 2]) < eps, eps, uvw[:, 2])

        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]
        z = P_cam[:, 2]
        return np.vstack((u, v, z)).T  # N×3

    def _bilinear_interpolate(self, img_f: np.ndarray, uv: np.ndarray) -> np.ndarray:
        """
        img_f: H×W×3 float32 в [0..1].
        uv:   M×2 вещественные pixel coords.
        Возвращает M×3 float32 в [0..1] (RGB).
        """
        h, w = img_f.shape[:2]
        u, v = uv[:,0], uv[:,1]

        x0 = np.floor(u).astype(int);  y0 = np.floor(v).astype(int)
        x1 = x0 + 1;                   y1 = y0 + 1

        # Кто вне границ?
        in_bounds = (x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)
        oob = np.count_nonzero(~in_bounds)
        if oob:
            logger.warning(f"[colorizer] {oob} точек за пределами → чёрный")

        # Clip индексы, чтобы не падать
        x0c = np.clip(x0, 0, w-1);  x1c = np.clip(x1, 0, w-1)
        y0c = np.clip(y0, 0, h-1);  y1c = np.clip(y1, 0, h-1)

        du = (u - x0).astype(np.float32)
        dv = (v - y0).astype(np.float32)

        wa = (1 - du)*(1 - dv)
        wb =    du *(1 - dv)
        wc = (1 - du)*    dv
        wd =    du *    dv

        Ia = img_f[y0c, x0c]
        Ib = img_f[y0c, x1c]
        Ic = img_f[y1c, x0c]
        Id = img_f[y1c, x1c]

        col = (Ia*wa[:,None] + Ib*wb[:,None] + Ic*wc[:,None] + Id*wd[:,None])
        # Если у кого–то вышли за границы — присвоим [0,0,0]
        if oob:
            col[~in_bounds] = 0.0
        # BGR→RGB
        return col[:, ::-1]

    def colorize(self, xyz: np.ndarray, img: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Основной метод. Возвращает цветной PointCloud.
        """
        # 1) Проецируем
        uvz = self.project_points(xyz)
        # 2) Фильтруем z>0 и в кадре
        h, w      = img.shape[:2]
        u, v, z   = uvz[:,0], uvz[:,1], uvz[:,2]
        mask      = (z>0) & (u>=0)&(u<w)&(v>=0)&(v<h)
        uv_vis    = uvz[mask, :2]
        xyz_vis   = xyz[mask]

        # 3) Нормализация изображения
        img_f = img.astype(np.float32) / 255.0

        # 4) Билинейное окрашивание
        colors = self._bilinear_interpolate(img_f, uv_vis)

        # 5) Сборка PointCloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.ascontiguousarray(xyz_vis))
        pcd.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(colors))
        return pcd
