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
    extract_roi_cloud,
    load_camera_params,
)

def adjust_corners_interactively(img, corners, pattern_size):
    """
    Открывает окно, в котором вы можете:
      - Кликнуть по любому кружку (углу) и, удерживая, перетащить
        его в нужное место (drag&drop).
      - Нажать Enter, чтобы зафиксировать результат.
    Возвращает скорректированный массив corners.
    """
    window = "Adjust corners (drag points, Enter to accept)"
    cv2.namedWindow(window)
    selected = -1  # индекс перетаскиваемой точки
    dragging = False
    radius = 10    # радиус зоны "хвата" в пикселях

    def on_mouse(ev, x, y, flags, param):
        nonlocal selected, dragging, corners
        if ev == cv2.EVENT_LBUTTONDOWN:
            # ищем ближайший кружок
            for i, (cx, cy) in enumerate(corners):
                if (x-cx)**2 + (y-cy)**2 < radius**2:
                    selected = i
                    dragging = True
                    break
        elif ev == cv2.EVENT_MOUSEMOVE and dragging and selected >= 0:
            # перетаскиваем выбранный кружок
            corners[selected] = [x, y]
        elif ev == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            selected = -1

    cv2.setMouseCallback(window, on_mouse)

    while True:
        vis = img.copy()
        # рамка ROI, если хотим
        # cv2.rectangle(vis, (x,y),(x+w,y+h),(0,255,0),2)
        cv2.drawChessboardCorners(vis, pattern_size,
                                  corners.reshape(-1,1,2), True)
        cv2.imshow(window, vis)
        key = cv2.waitKey(20)
        if key in (13, 10):  # Enter
            break

    cv2.destroyWindow(window)
    return corners


def main():
    parser = argparse.ArgumentParser(
        description="Полуавтомат: ROI-image + ручная правка 2D-углов + ROI-pointcloud"
    )
    parser.add_argument("-i","--image", required=True,
                        help="Путь к изображению (PNG/JPG)")
    parser.add_argument("-p","--pcd",   required=True,
                        help="Путь к облаку (.pcd/.ply/.bin/.txt)")
    parser.add_argument("-c","--calib", required=True,
                        help="Путь к calib_cam_to_cam.txt")
    parser.add_argument("-k","--camidx",type=int,default=0,
                        help="Индекс камеры 0–3")
    parser.add_argument("--pattern", nargs=2, type=int, default=[7,5],
                        help="cols rows внутренних углов")
    args = parser.parse_args()

    # 1) ROI на изображении
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError("Не удалось загрузить изображение")
    cols, rows = args.pattern
    pattern = (cols, rows)
    print(">>> Выберите ROI на изображении и нажмите ENTER")
    x, y, w, h = map(int, cv2.selectROI("Select ROI", img))
    cv2.destroyWindow("Select ROI")
    img_roi = img[y:y+h, x:x+w]

    # 2) Авто-детект 2D-углов
    print(f">>> Авто-детект {cols}×{rows} углов")
    corners_roi = detect_image_corners(img_roi, pattern)
    corners2d = corners_roi + np.array([x, y], dtype=np.float32)

    # 3) Ручная правка
    print(">>> Перетащите проблемные кружки, нажмите Enter")
    corners2d = adjust_corners_interactively(img, corners2d, pattern)

    # 4) Финальная проверка
    vis = img.copy()
    cv2.rectangle(vis, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.drawChessboardCorners(vis, pattern,
                              corners2d.reshape(-1,1,2), True)
    cv2.imshow("Final corners (read-only)", vis)
    print("Нажмите любую клавишу, если всё впорядке...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 5) ROI в облаке
    print(">>> ROI-pointcloud: F+LMB — pick/unpick, Q — finish")
    pcd = load_point_cloud(args.pcd)
    idx = select_pointcloud_roi(pcd)
    if not idx:
        print("Нет точек — выходим.")
        return
    board_roi, obb = extract_roi_cloud(pcd, idx, expand=0.001)

    # 6) Загрузка K, D
    K, D, _, _, Tr = load_camera_params(args.calib, args.camidx)
    print("K =\n", K)
    print("D =\n", D)

    # 7) Показываем облако + ROI
    lines = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    lines.paint_uniform_color((0,1,0))
    pcd.paint_uniform_color((0.7,0.7,0.7))
    board_roi.paint_uniform_color((1,0,0))
    vis3d = o3d.visualization.Visualizer()
    vis3d.create_window("Board ROI in point cloud")
    vis3d.add_geometry(pcd)
    vis3d.add_geometry(board_roi)
    vis3d.add_geometry(lines)
    opt = vis3d.get_render_option()
    opt.point_size = 5
    vis3d.run()
    vis3d.destroy_window()

    print("2D и 3D ROI готовы — можно генерировать 3D-точки и PnP.")

if __name__ == "__main__":
    main()
