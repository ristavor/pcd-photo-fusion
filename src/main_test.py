import argparse
import cv2
import numpy as np
from calibration.roi_selector import select_image_corners, load_point_cloud, select_pointcloud_corners
from rectifier import ImageRectifier

def main():
    parser = argparse.ArgumentParser(description="Интерактивный выбор 4 углов ROI")
    parser.add_argument("-i", "--image", required=True, help="Путь к искажённому изображению")
    parser.add_argument("-p", "--pcd", required=True, help="Путь к облаку (.pcd/.ply/.bin/.txt)")
    parser.add_argument("-c", "--calib", required=True, help="Путь к файлу calib_cam_to_cam.txt")
    parser.add_argument("-k", "--camidx", type=int, default=0, help="Индекс камеры для исправления искажений")
    args = parser.parse_args()

    # 1) Исправляем дисторсию
    rectifier = ImageRectifier(args.calib, args.camidx)
    img_raw = cv2.imread(args.image)
    img = rectifier.rectify(img_raw)

    # 2) Выбор 4 точек на изображении
    corners2d = select_image_corners(img)
    print("2D углы ROI:", corners2d.tolist())

    # 3) Загрузка point cloud и выбор 4 точек
    pcd = load_point_cloud(args.pcd)
    corners3d = select_pointcloud_corners(pcd)
    print("3D углы ROI:", corners3d.tolist())

if __name__ == "__main__":
    main()