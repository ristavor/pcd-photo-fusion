# File: calibration/roi_selector.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import open3d as o3d
from pathlib import Path


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
    ax.imshow(image[..., ::-1])  # BGR → RGB для matplotlib
    ax.set_title("ЛКМ: поставить до 4 точек, перетаскивание, Enter/дбл-клик — готово")

    def onselect(verts):
        nonlocal corners, finished
        if len(verts) < 4:
            print(f"Выбрано только {len(verts)} точек, нужно 4")
            return
        corners[:] = verts[:4]
        finished = True
        plt.close(fig)

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
    Пока что просто отображает облако точек в новом GUI.
    Пользователь может вращать и зумить, закрыть окно нажатием Q.
    В будущем сюда добавим логику выбора точек.
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    # инициализация приложения (singleton)
    app = gui.Application.instance
    app.initialize()

    # создаём окно
    window = app.create_window("3D PointCloud Viewer (Q — закрыть)", 800, 600)

    # создаём виджет сцены и прикрепляем к окну
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # добавляем point cloud на сцену
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    scene.scene.add_geometry("pcd", pcd, material)

    # подгоняем камеру
    bbox = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bbox, bbox.get_center())

    # обработчик клавиш — закроет окно по Q или q
    def on_key(event: gui.KeyEvent) -> bool:
        if event.type == gui.KeyEvent.Type.DOWN and \
                (event.key == gui.KeyName.Q or event.key == gui.KeyName.q):
            window.close()  # используем переменную window из внешней области видимости
            return True  # сигнализируем, что событие обработано
        return False  # передать событие дальше (необязательно)

    window.set_on_key(on_key)

    # запускаем GUI
    app.run()

    # пока возвращаем пустой массив
    return np.empty((0, 3))
