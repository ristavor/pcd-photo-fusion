"""
import cv2
import open3d as o3d
from pathlib import Path
import logging

from colorizer import (
    read_cam_to_cam, read_velo_to_cam,
    load_image, load_velodyne,
    Colorizer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    ROOT = Path(__file__).resolve().parent.parent

    # 1) Читаем калибровки
    cam_data = read_cam_to_cam(str(ROOT/'data/2011_09_28_calib/2011_09_28/calib_cam_to_cam.txt'))
    K        = cam_data['P_rect_02'].reshape(3,4)[:3,:3]
    R, T     = read_velo_to_cam(str(ROOT/'data/2011_09_28_calib/2011_09_28/calib_velo_to_cam.txt'))

    logger.info(f"K =\n{K}")
    logger.info(f"R =\n{R}")
    logger.info(f"T = {T}")

    # 2) Загружаем данные
    img   = load_image(str(ROOT/'data/2011_09_28_drive_0034_sync/image_02/data/0000000000.png'))
    pts4d = load_velodyne(str(ROOT/'data/2011_09_28_drive_0034_sync/velodyne_points/data/0000000000.bin'))
    xyz   = pts4d[:, :3]

    # 3) Покраска
    colorizer = Colorizer(R, T, K)
    pcd_colored = colorizer.colorize(xyz, img)

    # 4) Визуализация
    o3d.visualization.draw_geometries([pcd_colored],
                                      window_name='Colored',
                                      width=800, height=600)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
"""
#!/usr/bin/env python3
import sys
from pathlib import Path
import cv2

from src.rectifier.rectifier import ImageRectifier

def main():
    ROOT       = Path(__file__).resolve().parent.parent
    img_path   = ROOT / 'data/2011_09_28_drive_0034_extract/image_02/data/0000000000.png'
    calib_path = ROOT / 'data/2011_09_28_calib/calib_cam_to_cam.txt'

    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR loading image {img_path}", file=sys.stderr); sys.exit(1)

    h, w = img.shape[:2]

    cv2.imshow('Original', img)

    # запуск rectifier: камера #2, размер (w,h)
    rect = ImageRectifier(str(calib_path), cam_idx=2, image_size=(w,h))
    img_rect = rect.rectify(img)

    cv2.imshow('Rectified', img_rect)
    print("Press any key...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
