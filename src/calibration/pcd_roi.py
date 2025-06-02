import numpy as np
import open3d as o3d
from pathlib import Path
from typing import List, Tuple

# Можно перенести эту константу в utils/constants.py, если понадобится в других модулях.
SPHERE_RADIUS_FACTOR = 0.0001


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Загружает облако точек из файла.

    Поддерживаемые форматы:
      - .pcd, .ply: используем o3d.io.read_point_cloud.
      - .bin : Velodyne Binary → читаем float32, reshape(-1,4), берём первые три столбца.
      - .txt : текстовый файл XYZ или XYZI → np.loadtxt, берём первые три столбца.

    Параметры:
      path (str): путь к файлу облака точек.

    Возвращает:
      o3d.geometry.PointCloud: загруженное облако точек.

    Исключения:
      ValueError: если формат файла не поддерживается.
      RuntimeError: если облако пустое после загрузки.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext in ('.pcd', '.ply'):
        pcd = o3d.io.read_point_cloud(str(p))
    elif ext == '.bin':
        data = np.fromfile(str(p), dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(f"Неправильный формат Velodyne‐файла: {path}")
        data = data.reshape(-1, 4)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data[:, :3]))
    elif ext == '.txt':
        data = np.loadtxt(str(p), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"Текстовый файл {path} должен содержать минимум три столбца (X Y Z).")
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(data[:, :3]))
    else:
        raise ValueError(f"Неподдерживаемый формат облака точек: {ext}")

    if pcd.is_empty():
        raise RuntimeError(f"Загруженное облако точек пустое: {path}")
    return pcd


def select_pointcloud_roi(
    pcd: o3d.geometry.PointCloud,
    sphere_radius: float = None
) -> List[int]:
    """
    Запускает Open3D GUI для интерактивного выделения точек ROI (области интереса).

    Управление:
      - F + ЛКМ: отметить/снять отметку точки (появится/пропадёт красный шарик).
      - Q: завершить выбор и закрыть окно.

    Параметры:
      pcd (o3d.geometry.PointCloud): облако точек, из которого нужно выбрать ROI.
      sphere_radius (float, optional): радиус сферы визуализации выбранных точек.
        Если None, вычисляется как diag(bbox) * SPHERE_RADIUS_FACTOR.

    Возвращает:
      List[int]: список индексов точек, отмеченных пользователем.

    Примечания:
      - Используется Open3D Visualization GUI. Потребуются правильные зависимости.
      - Может быть медленно для очень больших облаков.
    """
    # Строим KD‐дерево один раз
    tree = o3d.geometry.KDTreeFlann(pcd)

    # Если радиус не передан, вычисляем по диагонали bounding box
    if sphere_radius is None:
        bbox = pcd.get_axis_aligned_bounding_box()
        diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        sphere_radius = diag * SPHERE_RADIUS_FACTOR

    selected = set()
    sphere_names = set()

    # Настройки материалов для облака и "шариков" выбранных точек
    mat_pcd = o3d.visualization.rendering.MaterialRecord()
    mat_pcd.shader = "defaultUnlit"
    mat_sel = o3d.visualization.rendering.MaterialRecord()
    mat_sel.shader = "defaultLit"
    mat_sel.base_color = (1.0, 0.0, 0.0, 1.0)  # красный

    # Инициализируем Open3D GUI
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance
    app.initialize()

    win = app.create_window("3D ROI: F+ЛКМ – выбрать/снять, Q – закончить", 800, 600)
    scene = gui.SceneWidget()
    scene.scene = rendering.Open3DScene(win.renderer)
    win.add_child(scene)

    # Добавляем исходное облако
    scene.scene.add_geometry("pcd", pcd, mat_pcd)

    # Настраиваем камеру, чтобы она охватила всё облако
    bbox = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bbox, bbox.get_center())

    def update_spheres() -> None:
        """
        Обновляет отрисовку красных сфёрок вокруг выбранных точек.
        Сначала удаляем все предыдущие сферы, затем добавляем
        по каждой точке из selected.
        """
        # Удаляем старые сферы
        for name in sphere_names:
            scene.scene.remove_geometry(name)
        sphere_names.clear()

        pts = np.asarray(pcd.points)
        for idx in selected:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sph.compute_vertex_normals()
            sph.translate(pts[idx])
            name = f"sel_{idx}"
            scene.scene.add_geometry(name, sph, mat_sel)
            sphere_names.add(name)

    def on_key(evt) -> bool:
        """
        Обработчик нажатий клавиш:
          - F (DOWN): включаем режим выбора (при зажатом F и клике ЛКМ отметка).
          - F (UP): выключаем режим выбора.
          - Q (DOWN): закрываем окно и завершаем выбор.
        """
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

    def on_mouse(evt) -> bool:
        """
        Обработчик кликов мыши:
          - Если в режиме выбора (picking=True) и событие BUTTON_DOWN ЛКМ:
              1) Считываем глубину в пикселе (x, y).
              2) Конвертируем (x, y, d) → мировые координаты (unproject).
              3) Находим ближайшую точку облака и переключаем её статус:
                 если уже была выбрана — снимаем, иначе — добавляем.
              4) Вызываем update_spheres() для перерисовки.
        """
        if not (picking and evt.type == gui.MouseEvent.Type.BUTTON_DOWN
                and evt.is_button_down(gui.MouseButton.LEFT)):
            return False

        x = evt.x - scene.frame.x
        y = evt.y - scene.frame.y
        if not (0 <= x < scene.frame.width and 0 <= y < scene.frame.height):
            return False

        def depth_callback(depth_image):
            d = np.asarray(depth_image)[y, x]
            # Игнорируем, если глубина >= 1.0 (нет интерпретации)
            if d >= 1.0:
                return
            # Получаем мировые координаты точки под курсором
            world = scene.scene.camera.unproject(
                x, y, d, scene.frame.width, scene.frame.height
            )
            # Ищем ближайшую точку в pcd
            _, idx_nn, _ = tree.search_knn_vector_3d(world, 1)
            idx0 = int(idx_nn[0])
            if idx0 in selected:
                selected.remove(idx0)
            else:
                selected.add(idx0)
            update_spheres()

        # Рендерим глубину и запускаем callback
        scene.scene.scene.render_to_depth_image(depth_callback)
        return True

    # Флаг, показывающий, нажата ли клавиша F (режим выбора)
    picking = False

    win.set_on_key(on_key)
    scene.set_on_mouse(on_mouse)

    app.run()

    # После закрытия окна возвращаем список выбранных индексов
    return list(selected)


def extract_roi_cloud(
    pcd: o3d.geometry.PointCloud,
    indices: List[int],
    expand: float = 0.01
) -> Tuple[o3d.geometry.PointCloud, o3d.geometry.OrientedBoundingBox]:
    """
    Строит ориентированный bounding box (OBB) вокруг точек с указанными индексами,
    расширяет его на expand по всем осям, и возвращает обрезанное облако и сам OBB.

    Параметры:
      pcd (o3d.geometry.PointCloud): исходное облако точек.
      indices (List[int]): индексы точек, вокруг которых строится OBB.
      expand (float): величина расширения OBB вдоль каждой оси (метры).

    Возвращает:
      cropped_cloud (o3d.geometry.PointCloud): облако, обрезанное по OBB.
      obb (o3d.geometry.OrientedBoundingBox): расширенный ориентированный ящик.

    Исключения:
      ValueError: если список индексов пуст или выходит за пределы облака.
    """
    if not indices:
        raise ValueError("Список индексов для extract_roi_cloud пуст.")
    points = np.asarray(pcd.points)
    if max(indices) >= len(points) or min(indices) < 0:
        raise ValueError("Некорректные индексы точек для extract_roi_cloud.")

    pts_subset = points[indices]
    obb = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts_subset))
    ext = obb.extent
    # Расширяем OBB по всем осям
    obb_expanded = o3d.geometry.OrientedBoundingBox(obb.center, obb.R, ext + expand)
    cropped_cloud = pcd.crop(obb_expanded)
    return cropped_cloud, obb_expanded
