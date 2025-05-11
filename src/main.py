#!/usr/bin/env python3
import cv2
import numpy as np
import open3d as o3d

from pathlib import Path
from src.synchronizer import Synchronizer
from src.rectifier       import ImageRectifier
from src.colorizer       import Colorizer

def load_velodyne_txt(path: Path) -> o3d.geometry.PointCloud:
    pts = np.loadtxt(str(path), dtype=np.float32)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    xyz = pts[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def load_velodyne_bin(path: Path) -> o3d.geometry.PointCloud:
    pts = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
    xyz = pts[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def main():
    # 1) Папки raw-extract и synced
    ROOT_extract = Path(__file__).resolve().parent.parent / "data" / "2011_09_28_drive_0034_extract"
    ROOT_sync    = Path(__file__).resolve().parent.parent / "data" / "2011_09_28_drive_0034_sync"

    # 2) Синхронизация
    sync = Synchronizer(raw_root=ROOT_extract, cam_folder="image_02")
    matches = sync.sync()
    if not matches:
        print("Synchronization failed — нет совпадений.")
        return

    # 3) Берём первую пару (0,0,0) после ренумерации
    i_cam, i_velo, _ = matches[0]

    # 4) Путь к сырым файлам
    img_raw_path  = ROOT_extract / "image_02"           / "data" / f"{i_cam:010d}.png"
    velo_raw_path = ROOT_extract / "velodyne_points"   / "data" / f"{i_velo:010d}.txt"

    # 5) Путь к synced‐файлам (для сравнения)
    img_sync_path  = ROOT_sync / "image_02"           / "data" / f"{i_cam:010d}.png"
    velo_sync_path = ROOT_sync / "velodyne_points"   / "data" / f"{i_velo:010d}.bin"

    # 6) Исправляем (rectify) изображение из raw, сравниваем с synced
    rectifier = ImageRectifier(
        calib_cam_path = Path(__file__).resolve().parent.parent
                          / "data" / "2011_09_28_calib" / "calib_cam_to_cam.txt",
        cam_idx = 2
    )
    img_rectified = rectifier.rectify(cv2.imread(str(img_raw_path), cv2.IMREAD_UNCHANGED))
    img_synced    = cv2.imread(str(img_sync_path), cv2.IMREAD_UNCHANGED)

    cv2.imshow("Raw → Rectified", img_rectified)
    cv2.imshow("KITTI-synced",    img_synced)

    # 7) Загружаем и красим облако из raw
    pcd_raw   = load_velodyne_txt(velo_raw_path)
    pcd_sync  = load_velodyne_bin(velo_sync_path)

    # читаем calib_velo_to_cam
    from src.colorizer import read_velo_to_cam
    R, T = read_velo_to_cam(
        Path(__file__).resolve().parent.parent / "data"
        / "2011_09_28_calib"/"calib_velo_to_cam.txt"
    )
    # проекция + окраска
    colorizer = Colorizer(R, T, rectifier.P_new)
    pcd_colored_raw  = colorizer.colorize(np.asarray(pcd_raw.points), img_rectified)
    pcd_colored_sync = colorizer.colorize(np.asarray(pcd_sync.points), img_synced)

    # 8) Показываем оба цветных облака в отдельных окнах
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window("Colored Raw→Rectified", 800, 600)
    vis1.add_geometry(pcd_colored_raw)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("Colored KITTI-synced", 800, 600)
    vis2.add_geometry(pcd_colored_sync)

    # главный цикл
    try:
        while True:
            cv2.waitKey(1)
            vis1.poll_events(); vis1.update_renderer()
            vis2.poll_events(); vis2.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        vis1.destroy_window()
        vis2.destroy_window()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
