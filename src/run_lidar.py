from pathlib import Path

import numpy as np
import open3d as o3d

# 1) Укажите путь к файлу облака точек
ROOT = Path(__file__).resolve().parent.parent
pcd_path = 'C:/Users/peamp/Downloads/kitti_txt.txt'


def load_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    """Load point cloud from .bin, .ply, or KITTI-style .txt file."""
    path = Path(path)
    
    if path.suffix == '.bin':
        # Load from Velodyne binary format
        pts = np.fromfile(str(path), dtype=np.float32)
        pts = pts.reshape(-1, 4)  # X, Y, Z, intensity
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3])
        return pcd
    elif path.suffix == '.ply':
        # Load directly using Open3D
        return o3d.io.read_point_cloud(str(path))
    elif path.suffix == '.txt':
        # Load KITTI-style txt: X Y Z R G B
        arr = np.loadtxt(str(path))
        if arr.shape[1] == 6:
            points = arr[:, :3]
            colors = arr[:, 3:6] / 255.0
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            return pcd
        else:
            raise ValueError("TXT file must have 6 columns: X Y Z R G B")
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


# 3) Загрузка облака и его визуализация
pcd = load_point_cloud(pcd_path)
print(f"Загружено точек: {len(pcd.points)}")
o3d.visualization.draw_geometries(
    [pcd],
    window_name='Point Cloud Viewer',
    width=800,
    height=600,
    point_show_normal=False
)
