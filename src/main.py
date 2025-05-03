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
import sys
from pathlib import Path

import cv2

from src.rectifier import ImageRectifier, read_cam_to_cam


def main():
    ROOT = Path(__file__).resolve().parent.parent

    # 1) Полный словарь калибровки
    calib_cam = ROOT / 'data' \
                / '2011_09_28_calib' \
                / 'calib_cam_to_cam.txt'
    cam_dict = read_cam_to_cam(str(calib_cam))

    # 2) Выбираем камеру №2
    cam_idx = 2
    rect = cam_dict[f'S_rect_0{cam_idx}'].astype(int)
    w_rect, h_rect = rect[0], rect[1]

    # 3) Загружаем исходник
    img_path = ROOT / 'data' \
               / '2011_09_28_drive_0034_extract' \
               / 'image_02' \
               / 'data' \
               / '0000000005.png'
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"ERROR: не удалось загрузить {img_path}", file=sys.stderr)
        sys.exit(1)
    cv2.imshow('Original', img)

    # 4) Делам rectify + crop
    rectifier = ImageRectifier(str(calib_cam), cam_idx)
    img_rect = rectifier.rectify(img)
    img_crop = img_rect[0:h_rect, 0:w_rect]
    cv2.imshow('Rectified & Cropped', img_crop)

    print("Нажмите любую клавишу для выхода…")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
