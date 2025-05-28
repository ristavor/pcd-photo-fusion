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
    compute_board_frame,
    generate_object_points
)

def adjust_corners_interactively(img, corners, pattern):
    """
    Drag&drop любого кружка:
      ЛКМ→захват, MOVE→движение, отпуск ЛКМ→финиш,
      Enter→окончательно.
    """
    win="Adjust corners"
    selected=-1; dragging=False; R=10
    cv2.namedWindow(win)
    def on_mouse(ev,x,y,flags,_):
        nonlocal selected,dragging,corners
        if ev==cv2.EVENT_LBUTTONDOWN:
            for i,(cx,cy) in enumerate(corners):
                if (x-cx)**2+(y-cy)**2<R*R:
                    selected=i; dragging=True; break
        elif ev==cv2.EVENT_MOUSEMOVE and dragging:
            corners[selected]=[x,y]
        elif ev==cv2.EVENT_LBUTTONUP and dragging:
            dragging=False; selected=-1
    cv2.setMouseCallback(win,on_mouse)
    while True:
        vis=img.copy()
        cv2.drawChessboardCorners(vis,pattern,corners.reshape(-1,1,2),True)
        cv2.imshow(win,vis)
        if cv2.waitKey(20) in (10,13): break
    cv2.destroyWindow(win)
    return corners

def main():
    p=argparse.ArgumentParser()
    p.add_argument("-i","--image", required=True)
    p.add_argument("-p","--pcd",   required=True)
    p.add_argument("-c","--calib", required=True)
    p.add_argument("-k","--camidx",type=int,default=0)
    p.add_argument("--pattern",nargs=2,type=int,default=[7,5])
    args=p.parse_args()

    # 1) ROI на изображении
    img=cv2.imread(args.image)
    cols,rows=args.pattern; pattern=(cols,rows)
    print("Select ROI → ENTER")
    x,y,w,h=map(int,cv2.selectROI("ROI image",img))
    cv2.destroyWindow("ROI image")
    img_roi=img[y:y+h, x:x+w]

    # 2) Авто-детект 2D-углов
    print("Auto-detect corners")
    corners_roi=detect_image_corners(img_roi,pattern)
    corners2d=corners_roi+np.array([x,y],dtype=np.float32)

    # 3) Ручная правка
    print("Adjust corners → ENTER")
    corners2d=adjust_corners_interactively(img,corners2d,pattern)

    # 4) Финальный read-only вывод кружков
    vis=img.copy()
    cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.drawChessboardCorners(vis,pattern,corners2d.reshape(-1,1,2),True)
    cv2.imshow("Final corners",vis)
    print("OK? any key → continue")
    cv2.waitKey(0); cv2.destroyAllWindows()

    # 5) ROI в облаке
    print("Select cloud ROI: F+LMB pick/unpick, Q finish")
    pcd=load_point_cloud(args.pcd)
    ids=select_pointcloud_roi(pcd)
    if not ids:
        print("No points — abort"); return
    board_roi, obb = extract_roi_cloud(pcd, ids, expand=0.001)

    # 6) Локальная СК доски
    origin, x_axis, y_axis, normal = compute_board_frame(board_roi)
    print("Board frame:",origin, x_axis, y_axis, normal)

    # 7) Визуализация облака + ROI + оси
    # OBB каркас
    obb_ls=o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
    obb_ls.paint_uniform_color([0,1,0])
    # оси
    pts=[origin,
         origin+x_axis*0.2,
         origin+y_axis*0.2,
         origin+normal*0.2]
    pts = [
        origin,
        origin + x_axis * 0.2,  # X-ось
        origin + y_axis * 0.2,  # Y-ось
        origin + normal * 0.2  # Z-ось
    ]
    axes = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])
    )
    # Теперь задаём цвета рёбер
    axes.colors = o3d.utility.Vector3dVector([
        [1.0, 0.0, 0.0],  # красная X-ось
        [0.0, 1.0, 0.0],  # зелёная Y-ось
        [0.0, 0.0, 1.0],  # синяя Z-ось
    ])
    pcd.paint_uniform_color([0.7,0.7,0.7])
    board_roi.paint_uniform_color([1,0,0])

    vis3d=o3d.visualization.Visualizer(); vis3d.create_window("Cloud+ROI+Axes")
    vis3d.add_geometry(pcd); vis3d.add_geometry(board_roi)
    vis3d.add_geometry(obb_ls); vis3d.add_geometry(axes)
    opt=vis3d.get_render_option(); opt.point_size=3
    vis3d.run(); vis3d.destroy_window()

    print("Pipeline complete: image ROI, corners, cloud ROI, axes.")
    # 8) Генерация «идеальных» 3D-углов шахматной доски
    square_size = 0.10  # метр — размер клетки шахматной доски в KITTI
    cols, rows = pattern
    half_offset = ((cols - 1) / 2) * square_size * x_axis \
                  + ((rows - 1) / 2) * square_size * y_axis
    origin0 = origin - half_offset
    print("Adjusted origin for grid corner:", origin0)
    corners3d = generate_object_points(
        origin0, x_axis, y_axis,
        pattern, square_size
    )
    print(f"Generated {len(corners3d)} ideal 3D corners.")
    # Визуализация: жёлтые точки поверх board_roi
    obj_pcd = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(corners3d)
    )
    obj_pcd.paint_uniform_color([1.0, 1.0, 0.0])  # жёлтые
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window("Ideal 3D corners", width=800, height=600)
    # Показываем тот же board_roi + corners
    vis2.add_geometry(board_roi)
    vis2.add_geometry(obj_pcd)
    opt2 = vis2.get_render_option()
    opt2.point_size = 8
    opt2.background_color = np.array([0.1, 0.1, 0.1])
    vis2.run()
    vis2.destroy_window()
    print("Step 8 complete: ideal 3D corners generated and visualized.")


if __name__=="__main__":
    main()
