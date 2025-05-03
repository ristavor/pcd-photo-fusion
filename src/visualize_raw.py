from pathlib import Path

import numpy as np
import open3d as o3d

# 1) Укажите путь к бинарному файлу Velodyne (берём кадр 0000000000)
ROOT = Path(__file__).resolve().parent.parent
velo_path = ROOT / 'data/2011_09_28_drive_0034_sync' \
            / 'velodyne_points' / 'data' / '0000000000.bin'


# 2) Функция загрузки LiDAR точек из .bin
def load_velodyne_bin(path: Path) -> o3d.geometry.PointCloud:
    pts = np.fromfile(str(path), dtype=np.float32)
    pts = pts.reshape(-1, 4)  # X, Y, Z, intensity
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
    return pcd


# 3) Загрузка облака и его визуализация
pcd = load_velodyne_bin(velo_path)
print(f"Загружено точек: {len(pcd.points)}")
o3d.visualization.draw_geometries(
    [pcd],
    window_name='Raw LiDAR Point Cloud',
    width=800,
    height=600,
    point_show_normal=False
)
