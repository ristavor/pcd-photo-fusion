# File: calibration/roi_selector.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import open3d as o3d
from pathlib import Path
import traceback  # для печати трассировки ошибок


def select_image_corners(image: np.ndarray) -> np.ndarray:
    """
    Интерактивный выбор ровно 4 углов шахматной доски на изображении:
      - ЛКМ: добавить вершину (до 4),
      - Перетаскивание вершины: захватить и двигать,
      - Enter или двойной клик: завершить выбор.
    Возвращает np.ndarray shape (4,2) с координатами (x, y) в пикселях.
    """
    corners = []
    finished = False

    fig, ax = plt.subplots()
    ax.imshow(image[..., ::-1])  # BGR → RGB
    ax.set_title("ЛКМ: поставить до 4 точек, перетаскивание, Enter/дбл-клик — готово")

    def onselect(verts):
        nonlocal corners, finished
        try:
            if len(verts) < 4:
                print(f"Выбрано только {len(verts)} точек, нужно 4")
                return
            corners[:] = verts[:4]
            finished = True
            plt.close(fig)
        except Exception:
            print("Ошибка в onselect (2D):")
            traceback.print_exc()

    selector = PolygonSelector(
        ax, onselect,
        useblit=True,
        props=dict(color='r', linestyle='-', linewidth=2),
        handle_props=dict(
            marker='o', markersize=8,
            markeredgecolor='r', markerfacecolor='r',
            linestyle='None'
        ),
        grab_range=5,
        draw_bounding_box=False
    )

    plt.show()

    if not finished or len(corners) != 4:
        raise ValueError(f"Нужно выбрать ровно 4 угла, выбрано: {len(corners)}")

    return np.array(corners, dtype=np.int32)


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Загружает облако точек из .pcd/.ply, .bin (Velodyne) или .txt (ASCII).
    Возвращает open3d.geometry.PointCloud.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in ('.pcd', '.ply'):
        pcd = o3d.io.read_point_cloud(str(p))

    elif ext == '.bin':
        data = np.fromfile(str(p), dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(f"Неправильный .bin формат: {path}")
        pts = data.reshape(-1, 4)[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    elif ext == '.txt':
        data = np.loadtxt(str(p), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"В .txt должно быть ≥3 колонки XYZ: {path}")
        pts = data[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))

    else:
        raise ValueError(f"Неподдерживаемый формат облака: {ext}")

    if pcd.is_empty():
        raise ValueError(f"Облако пусто или некорректно: {path}")

    return pcd


def select_pointcloud_corners(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    Отображает облако точек в собственном GUI.
    F+ЛКМ: выбрать/снять точку; Q: закрыть окно.
    Возвращает координаты выбранных точек (M×3).
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    # Инициализация приложения
    app = gui.Application.instance
    app.initialize()
    window = app.create_window("3D PointCloud Viewer (F+LMB to pick, Q to quit)", 800, 600)

    # Сцена и основное облако
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    mat_base = rendering.MaterialRecord()
    mat_base.shader = "defaultUnlit"
    scene.scene.add_geometry("pcd", pcd, mat_base)

    # Настраиваем камеру
    bbox = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bbox, bbox.get_center())

    # Облачко для выбранных точек
    sel_pcd = o3d.geometry.PointCloud()
    mat_sel = rendering.MaterialRecord()
    mat_sel.shader = "defaultUnlit"
    mat_sel.base_color = (1, 0, 0, 1.0)
    scene.scene.add_geometry("selected", sel_pcd, mat_sel)

    flann = o3d.geometry.KDTreeFlann(pcd)
    f_down = False
    selected_idx = set()

    # Клавиатурный колбэк
    def _on_key(event: gui.KeyEvent) -> bool:
        nonlocal f_down
        try:
            if event.type == gui.KeyEvent.Type.DOWN and event.key == gui.KeyName.F:
                f_down = True
                return True
            if event.type == gui.KeyEvent.Type.UP and event.key == gui.KeyName.F:
                f_down = False
                return True
            if event.type == gui.KeyEvent.Type.DOWN and event.key == gui.KeyName.Q:
                window.close()
                return True
        except Exception:
            print("Ошибка в on_key (3D):")
            traceback.print_exc()
        return False

    window.set_on_key(_on_key)

    # Мышиный колбэк
    def _on_mouse(mouse_evt: gui.MouseEvent) -> bool:
        nonlocal selected_idx
        try:
            if not (f_down
                    and mouse_evt.type == gui.MouseEvent.Type.BUTTON_DOWN
                    and mouse_evt.is_button_down(gui.MouseButton.LEFT)):
                return False

            x = mouse_evt.x - scene.frame.x
            y = mouse_evt.y - scene.frame.y
            if x < 0 or y < 0 or x >= scene.frame.width or y >= scene.frame.height:
                return False

            def _depth_cb(depth_image):
                try:
                    depth = np.asarray(depth_image)[y, x]
                    if depth == 1.0:
                        return
                    world = scene.scene.camera.unproject(
                        x, y, depth, scene.frame.width, scene.frame.height)
                    k, idx, _ = flann.search_knn_vector_3d(world, 1)
                    if k == 0:
                        return
                    i0 = idx[0]
                    if i0 in selected_idx:
                        selected_idx.remove(i0)
                    else:
                        selected_idx.add(i0)
                    # обновляем облачко выбранных точек
                    pts = np.asarray(pcd.points)[list(selected_idx)]
                    sel_pcd.points = o3d.utility.Vector3dVector(pts)
                    # планируем обновление из основного потока
                    def _upd():
                        scene.scene.remove_geometry("selected")
                        scene.scene.add_geometry("selected", sel_pcd, mat_sel)
                    gui.Application.instance.post_to_main_thread(window, _upd)
                except Exception:
                    print("Ошибка в depth callback:")
                    traceback.print_exc()

            # рендерим глубину через низкоуровневый rendering.Scene
            scene.scene.scene.render_to_depth_image(_depth_cb)
            return True

        except Exception:
            print("Ошибка в on_mouse (3D):")
            traceback.print_exc()
            return False

    scene.set_on_mouse(_on_mouse)

    # Запускаем GUI
    try:
        app.run()
    except Exception:
        print("Ошибка при запуске GUI:")
        traceback.print_exc()

    # Возвращаем выбранные 3D-точки
    return np.asarray(pcd.points)[list(selected_idx)]
