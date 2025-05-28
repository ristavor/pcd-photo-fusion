# File: calibration/roi_selector.py

import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from utils.calib import read_kitti_cam_calib, read_velo_to_cam

SPHERE_RADIUS_FACTOR = 0.0001

def detect_image_corners(image: np.ndarray, pattern_size=(7,5)) -> np.ndarray:
    """
    Автоматически находит все внутренние углы шахматной доски в image.
    Возвращает массив shape (N,2) с пиксельными координатами.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise RuntimeError(f"[ROI_SELECTOR] Не удалось найти {pattern_size} углов")
    cv2.cornerSubPix(
        gray, corners, winSize=(11,11), zeroZone=(-1,-1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    )
    return corners.reshape(-1,2).astype(np.float32)


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Загружает облако точек из .pcd/.ply/.bin/.txt.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in ('.pcd', '.ply'):
        pcd = o3d.io.read_point_cloud(str(p))
    elif ext == '.bin':
        data = np.fromfile(str(p), np.float32)
        pts = data.reshape(-1,4)[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    elif ext == '.txt':
        data = np.loadtxt(str(p), np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        pts = data[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    else:
        raise ValueError(f"[ROI_SELECTOR] Неподдерживаемый формат: {ext}")
    if pcd.is_empty():
        raise ValueError(f"[ROI_SELECTOR] Пустое облако: {path}")
    return pcd


def select_pointcloud_roi(
    pcd: o3d.geometry.PointCloud,
    sphere_radius: float = None
) -> list[int]:
    """
    Интерактивный выбор точек LiDAR-облака, принадлежащих шахматной доске.
    F + ЛКМ — переключить точку (добавить/снять);
    Q — закончить выбор.
    Возвращает список индексов выбранных точек.
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()
    window = app.create_window("3D: F+LMB — pick/unpick, Q — finish", 800, 600)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # базовое облако
    mat_base = rendering.MaterialRecord()
    mat_base.shader = "defaultUnlit"
    scene.scene.add_geometry("pcd", pcd, mat_base)

    # камера
    bbox = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bbox, bbox.get_center())

    # сферы для выбранных точек
    if sphere_radius is None:
        diag = np.linalg.norm(np.asarray(bbox.get_max_bound()) - np.asarray(bbox.get_min_bound()))
        sphere_radius = diag * SPHERE_RADIUS_FACTOR
    mat_sphere = rendering.MaterialRecord()
    mat_sphere.shader = "defaultLit"
    mat_sphere.base_color = (1.0, 0.0, 0.0, 1.0)

    flann = o3d.geometry.KDTreeFlann(pcd)
    picking = False
    selected_idx = set()
    sphere_names = set()

    def update_spheres():
        for name in sphere_names:
            scene.scene.remove_geometry(name)
        sphere_names.clear()
        pts = np.asarray(pcd.points)
        for i in selected_idx:
            center = pts[i]
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sph.compute_vertex_normals()
            sph.translate(center)
            name = f"sel_sph_{i}"
            scene.scene.add_geometry(name, sph, mat_sphere)
            sphere_names.add(name)

    def on_key(ev: gui.KeyEvent) -> bool:
        nonlocal picking
        if ev.type == gui.KeyEvent.Type.DOWN:
            if ev.key == gui.KeyName.F:
                picking = True
                return True
            if ev.key == gui.KeyName.Q:
                window.close()
                return True
        if ev.type == gui.KeyEvent.Type.UP and ev.key == gui.KeyName.F:
            picking = False
            return True
        return False

    window.set_on_key(on_key)

    def on_mouse(ev: gui.MouseEvent) -> bool:
        if not (picking
                and ev.type == gui.MouseEvent.Type.BUTTON_DOWN
                and ev.is_button_down(gui.MouseButton.LEFT)):
            return False

        x = ev.x - scene.frame.x
        y = ev.y - scene.frame.y
        if not (0 <= x < scene.frame.width and 0 <= y < scene.frame.height):
            return False

        def depth_cb(depth_image):
            d = np.asarray(depth_image)[y, x]
            if d >= 1.0:
                return
            world = scene.scene.camera.unproject(
                x, y, d,
                scene.frame.width, scene.frame.height)
            _, idx, _ = flann.search_knn_vector_3d(world, 1)
            i0 = idx[0]
            if i0 in selected_idx:
                selected_idx.remove(i0)
            else:
                selected_idx.add(i0)
            update_spheres()

        scene.scene.scene.render_to_depth_image(depth_cb)
        return True

    scene.set_on_mouse(on_mouse)

    app.run()
    return list(selected_idx)


def extract_roi_cloud(
    pcd: o3d.geometry.PointCloud,
    indices: list[int],
    expand: float = 0.01
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.OrientedBoundingBox]:
    """
    Строит OrientedBoundingBox по точкам indices и обрезает pcd.
    expand — добавить рамке небольшой запас (м).
    Возвращает (cropped_cloud, obb).
    """
    pts = np.asarray(pcd.points)[indices]
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(pts))
    old_ext = obb.extent
    new_ext = old_ext + np.array([expand, expand, expand], dtype=old_ext.dtype)
    obb = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, new_ext)
    cropped = pcd.crop(obb)
    return cropped, obb
