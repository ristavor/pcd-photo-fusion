#!/usr/bin/env python3
# File: src/main_test.py

import argparse
import cv2
import numpy as np
import open3d as o3d

from calibration.roi_selector import (
    detect_image_corners,
    load_point_cloud,
    select_pointcloud_roi,
    extract_roi_cloud
)

def main():
    parser = argparse.ArgumentParser(
        description="ROI-image + ROI-pointcloud для шахматной доски"
    )
    parser.add_argument("-i", "--image", required=True,
                        help="Путь к исходному изображению (PNG/JPG)")
    parser.add_argument("-p", "--pcd", required=True,
                        help="Путь к облаку LiDAR (.pcd/.ply/.bin/.txt)")
    parser.add_argument("--pattern", nargs=2, type=int,
                        default=[7, 5],
                        help="Число внутренних углов доски: cols rows")
    args = parser.parse_args()

    # --- Image ROI + corner detection ---
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {args.image}")

    cols, rows = args.pattern
    pattern_size = (cols, rows)

    print(">>> Выберите ROI на изображении и нажмите ENTER")
    roi = cv2.selectROI("Select ROI", img)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        print("ROI не выбран, выходим.")
        return

    img_roi = img[y:y+h, x:x+w]
    print(f">>> Детектируем {cols}×{rows} углов внутри ROI")
    corners_roi = detect_image_corners(img_roi, pattern_size)
    corners2d = corners_roi + np.array([x, y], np.float32)

    # Показ результата
    vis = img.copy()
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.drawChessboardCorners(vis, pattern_size,
                              corners2d.reshape(-1,1,2), True)
    cv2.imshow("Detected corners", vis)
    print("Нажмите любую клавишу, чтобы продолжить...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # --- Pointcloud ROI ---
    print(">>> Загружаем облако и запускаем выбор точек доски (F+LMB, Q)")
    pcd = load_point_cloud(args.pcd)
    indices = select_pointcloud_roi(pcd)
    if not indices:
        print("Точки не выбраны, выходим.")
        return

    # Обрезаем облако по ROI и получаем OBB
    board_roi, obb = extract_roi_cloud(pcd, indices, expand=0.001)

    # Подготовим LineSet для отображения границ OBB
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lines.paint_uniform_color((0.0, 1.0, 0.0))  # зелёный каркас

    # Покрасим облака
    board_roi.paint_uniform_color([1.0, 0.0, 0.0])
    pcd.paint_uniform_color([0.7, 0.7, 0.7])

    # Визуализация с увеличенным размером точек
    vis_o3d = o3d.visualization.Visualizer()
    vis_o3d.create_window(window_name="Board ROI in point cloud", width=800, height=600)
    vis_o3d.add_geometry(pcd)
    vis_o3d.add_geometry(board_roi)
    vis_o3d.add_geometry(lines)
    opt = vis_o3d.get_render_option()
    opt.point_size = 5
    opt.background_color = np.array([0.1, 0.1, 0.1])
    vis_o3d.run()
    vis_o3d.destroy_window()

    print("Область доски выделена. Далее можно генерировать 3D-углы и считать R/T.")

if __name__ == "__main__":
    main()
