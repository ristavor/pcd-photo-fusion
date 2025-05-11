import logging

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

from utils.constants import EPS_DIV, BILINEAR_OFFSETS

logger = logging.getLogger(__name__)


class Colorizer:
    """
    Окрашивает LiDAR-облако точек цветами из картинки.
    """

    def __init__(self, R: NDArray[np.float64], T: NDArray[np.float64], K: NDArray[np.float64]) -> None:
        # Проверяем формы матриц
        assert isinstance(R, np.ndarray) and R.shape == (3, 3), "R must be 3×3 numpy array"
        assert isinstance(T, np.ndarray) and T.shape == (3,), "T must be length-3 numpy array"
        assert isinstance(K, np.ndarray) and K.shape == (3, 3), "K must be 3×3 numpy array"
        self.R = R
        self.T = T
        self.K = K

    def project_points(self, pts: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Проектирует LiDAR-точки в камеру.
        :param pts: (N×3) XYZ LiDAR-поинты.
        :return: uvz (N×3) — [u, v, z_cam].
        """
        assert isinstance(pts, np.ndarray), "pts must be numpy array"
        assert pts.ndim == 2 and pts.shape[1] == 3, "pts must have shape (N,3)"

        P_cam = (self.R @ pts.T + self.T[:, None]).T  # (N,3)
        uvw = (self.K @ P_cam.T).T  # (N,3)

        # Защита от деления на ноль
        uvw[:, 2] = np.where(np.abs(uvw[:, 2]) < EPS_DIV, EPS_DIV, uvw[:, 2])

        u = uvw[:, 0] / uvw[:, 2]
        v = uvw[:, 1] / uvw[:, 2]
        z = P_cam[:, 2]
        return np.vstack((u, v, z)).T  # (N,3)

    def _bilinear_interpolate(self, img_f: NDArray[np.float32], uv: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Билинейная интерполяция по четырём соседним пикселям.
        :param img_f: H×W×3 float32 в [0..1]
        :param uv:     M×2 вещественные pixel coords
        :return:       M×3 float32 в [0..1] (RGB)
        """
        assert img_f.ndim == 3 and img_f.shape[2] == 3, "img_f must be H×W×3 float32 array"
        assert uv.ndim == 2 and uv.shape[1] == 2, "uv must have shape (M,2)"

        h, w = img_f.shape[:2]
        u, v = uv[:, 0], uv[:, 1]

        x0 = np.floor(u).astype(int)
        y0 = np.floor(v).astype(int)

        # смещения для 4 точек
        coords = []
        for dx, dy in BILINEAR_OFFSETS:
            xs = np.clip(x0 + dx, 0, w - 1)
            ys = np.clip(y0 + dy, 0, h - 1)
            coords.append((xs, ys))
        Ia = img_f[coords[0][1], coords[0][0]]
        Ib = img_f[coords[1][1], coords[1][0]]
        Ic = img_f[coords[2][1], coords[2][0]]
        Id = img_f[coords[3][1], coords[3][0]]

        du = (u - x0).astype(np.float32)
        dv = (v - y0).astype(np.float32)
        wa = (1 - du) * (1 - dv)
        wb = du * (1 - dv)
        wc = (1 - du) * dv
        wd = du * dv

        col = (Ia * wa[:, None] +
               Ib * wb[:, None] +
               Ic * wc[:, None] +
               Id * wd[:, None])

        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.all(in_bounds):
            oob = np.count_nonzero(~in_bounds)
            logger.warning(f"[Colorizer] {oob} points out-of-bounds, coloring them black")
            col[~in_bounds] = 0.0

        # BGR → RGB
        return col[:, ::-1]

    def colorize(self, xyz: NDArray[np.float64], img: NDArray[np.uint8]) -> o3d.geometry.PointCloud:
        """
        Окрашивает облако xyz цветами из img.
        """
        assert isinstance(xyz, np.ndarray) and xyz.ndim == 2 and xyz.shape[1] == 3, \
            "xyz must be numpy array of shape (N,3)"
        assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] in (3, 4), \
            "img must be H×W×3 or H×W×4 BGR image"

        uvz = self.project_points(xyz)
        h, w = img.shape[:2]
        u, v, z = uvz[:, 0], uvz[:, 1], uvz[:, 2]

        mask = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        uv_vis = uvz[mask, :2]
        xyz_vis = xyz[mask]

        img_f = img.astype(np.float32) / 255.0
        colors = self._bilinear_interpolate(img_f, uv_vis)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_vis)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
