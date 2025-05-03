import cv2
import open3d as o3d
from pathlib import Path

from src.calibrator import read_cam_to_cam, read_velo_to_cam
from src.loader     import load_image, load_velodyne
from src.colorize   import colorize

def main():
    # --- Инициализация путей и чтение калибровок ---
    ROOT = Path(__file__).resolve().parent.parent

    cam_calib  = read_cam_to_cam(ROOT / 'data/2011_09_28_calib/2011_09_28/calib_cam_to_cam.txt')
    K          = cam_calib['P_rect_02'].reshape(3,4)[:3, :3]

    R, T = read_velo_to_cam(ROOT / 'data/2011_09_28_calib/2011_09_28/calib_velo_to_cam.txt')
    print("K =", K)
    print("R =", R)
    print("T =", T)

    # --- Пути к данным ---
    img_path  = ROOT / 'data/2011_09_28_drive_0034_sync/image_02/data/0000000000.png'
    velo_path = ROOT / 'data/2011_09_28_drive_0034_sync/velodyne_points/data/0000000000.bin'

    # --- Загрузка ---
    img = load_image(img_path)
    pts = load_velodyne(velo_path)
    print(f"Image: shape={img.shape}, dtype={img.dtype}")
    print(f"Point cloud: {pts.shape[0]} points, first 5:\n{pts[:5]}")

    xyz = pts[:, :3]  # отбрасываем intensity

    # --- Окрашиваем облако ---
    pcd_colored = colorize(xyz, img, R, T, K)

    # --- Визуализация ---
    o3d.visualization.draw_geometries(
        [pcd_colored],
        window_name='Colored Point Cloud',
        width=800,
        height=600
    )

    print("Закройте окно 'Colored Point Cloud', затем нажмите Enter для выхода.")
    input()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
