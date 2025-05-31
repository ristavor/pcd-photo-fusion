# calibration/roi_selector/pcd_roi.py

import numpy as np
import open3d as o3d
from pathlib import Path


SPHERE_RADIUS_FACTOR = 0.0001


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Загружает файл облака точек:
      - .pcd, .ply  → o3d.io.read_point_cloud
      - .bin (.bin Velodyne) → читаем float32, reshape(-1,4), берём XYZ
      - .txt (x y z intensity) → np.loadtxt, берём только XYZ
    Возвращает open3d.geometry.PointCloud.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in ('.pcd', '.ply'):
        pcd = o3d.io.read_point_cloud(str(p))
    elif ext == '.bin':
        data = np.fromfile(str(p), dtype=np.float32).reshape(-1, 4)
        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(data[:, :3])
        )
    elif ext == '.txt':
        data = np.loadtxt(str(p), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        pcd = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(data[:, :3])
        )
    else:
        raise ValueError(f"Неподдерживаемый формат точек: {ext}")

    if pcd.is_empty():
        raise RuntimeError(f"Пустое облако: {path}")
    return pcd


def select_pointcloud_roi(
    pcd: o3d.geometry.PointCloud,
    sphere_radius: float = None
) -> list[int]:
    """
    Запускает Open3D GUI, позволяет пользователю маркировать точки
    (F + ЛКМ — отмечать/снимать отметку, Q — закончить выбор).
    После чего возвращает список индексов выбранных точек.
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()

    win = app.create_window("3D ROI: F+LMB pick / Q finish", 800, 600)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    win.add_child(scene)

    # Рендерим само облако
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    scene.scene.add_geometry("pcd", pcd, mat)

    # Камера смотрит на весь бокс
    bbox = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bbox, bbox.get_center())

    # Выбираем радиус для сферы (если не задан, считаем по диагонали bbox)
    if sphere_radius is None:
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        sphere_radius = diag * SPHERE_RADIUS_FACTOR

    mat_s = rendering.MaterialRecord()
    mat_s.shader = "defaultLit"
    mat_s.base_color = (1.0, 0.0, 0.0, 1.0)  # красные шарики

    flann = o3d.geometry.KDTreeFlann(pcd)
    selected = set()
    sphere_names = set()
    picking = False

    def update_spheres():
        # Сначала удаляем старые сферы
        for name in sphere_names:
            scene.scene.remove_geometry(name)
        sphere_names.clear()

        pts = np.asarray(pcd.points)
        for idx in selected:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sph.compute_vertex_normals()
            sph.translate(pts[idx])
            name = f"sel_{idx}"
            scene.scene.add_geometry(name, sph, mat_s)
            sphere_names.add(name)

    def on_key(evt):
        nonlocal picking
        if evt.type == gui.KeyEvent.Type.DOWN:
            if evt.key == gui.KeyName.F:
                picking = True
                return True
            if evt.key == gui.KeyName.Q:
                win.close()
                return True
        if evt.type == gui.KeyEvent.Type.UP and evt.key == gui.KeyName.F:
            picking = False
            return True
        return False

    def on_mouse(evt):
        if not (picking and evt.type == gui.MouseEvent.Type.BUTTON_DOWN
                and evt.is_button_down(gui.MouseButton.LEFT)):
            return False
        x = evt.x - scene.frame.x
        y = evt.y - scene.frame.y
        if not (0 <= x < scene.frame.width and 0 <= y < scene.frame.height):
            return False

        def depth_cb(depth):
            d = np.asarray(depth)[y, x]
            if d >= 1.0:
                return
            world = scene.scene.camera.unproject(
                x, y, d, scene.frame.width, scene.frame.height
            )
            _, idx, _ = flann.search_knn_vector_3d(world, 1)
            i0 = int(idx[0])
            if i0 in selected:
                selected.remove(i0)
            else:
                selected.add(i0)
            update_spheres()

        scene.scene.scene.render_to_depth_image(depth_cb)
        return True

    win.set_on_key(on_key)
    scene.set_on_mouse(on_mouse)

    app.run()
    return list(selected)


def extract_roi_cloud(
    pcd: o3d.geometry.PointCloud,
    indices: list[int],
    expand: float = 0.01
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.OrientedBoundingBox]:
    """
    По списку индексов точек строит OrientedBoundingBox вокруг них,
    расширяет его на expand (везде), и возвращает (обрезанное богатое облако, обрезанный OBB).
    """
    pts = np.asarray(pcd.points)[indices]
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(pts)
    )
    ext = obb.extent
    obb = o3d.geometry.OrientedBoundingBox(
        obb.center, obb.R, ext + expand
    )
    return pcd.crop(obb), obb
