# File: calibration/roi_selector.py

import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from utils.calib import read_kitti_cam_calib, read_velo_to_cam

SPHERE_RADIUS_FACTOR = 0.0001

def detect_image_corners(image: np.ndarray, pattern_size=(7,5)) -> np.ndarray:
    """
    Авто-детект всех внутренних углов шахматной доски + точная субпиксельная
    доработка. Возвращает ndarray (N,2).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH|cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise RuntimeError(f"[ROI_SELECTOR] Не удалось найти {pattern_size} углов")
    cv2.cornerSubPix(
        gray, corners,
        winSize=(11,11), zeroZone=(-1,-1),
        criteria=(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    )
    return corners.reshape(-1,2).astype(np.float32)


def load_point_cloud(path: str) -> o3d.geometry.PointCloud:
    """
    Загружает .pcd/.ply/.bin/.txt → o3d.geometry.PointCloud
    """
    p = Path(path); ext = p.suffix.lower()
    if ext in ('.pcd','.ply'):
        pcd = o3d.io.read_point_cloud(str(p))
    elif ext == '.bin':
        data = np.fromfile(str(p), np.float32)
        pts = data.reshape(-1,4)[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    elif ext == '.txt':
        data = np.loadtxt(str(p), np.float32)
        if data.ndim==1: data = data.reshape(1,-1)
        pts = data[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    else:
        raise ValueError(f"[ROI_SELECTOR] Неподдерживаемый формат {ext}")
    if pcd.is_empty():
        raise ValueError(f"[ROI_SELECTOR] Пустое облако: {path}")
    return pcd


def select_pointcloud_roi(
    pcd: o3d.geometry.PointCloud,
    sphere_radius: float=None
) -> list[int]:
    """
    GUI-интерфейс (F+ЛКМ to pick/unpick, Q to finish) для выбора индексов точек доски.
    """
    import open3d.visualization.gui as gui
    import open3d.visualization.rendering as rendering

    app = gui.Application.instance; app.initialize()
    window = app.create_window("3D ROI: F+LMB pick/unpick, Q finish",800,600)
    scene = gui.SceneWidget(); scene.scene=rendering.Open3DScene(window.renderer)
    window.add_child(scene)

    # показать весь LiDAR
    mat = rendering.MaterialRecord(); mat.shader="defaultUnlit"
    scene.scene.add_geometry("pcd", pcd, mat)
    bbox = pcd.get_axis_aligned_bounding_box()
    scene.setup_camera(60.0, bbox, bbox.get_center())

    # настроить радиус для сферы-маркера
    if sphere_radius is None:
        diag = np.linalg.norm(np.asarray(bbox.get_max_bound())-np.asarray(bbox.get_min_bound()))
        sphere_radius = diag * SPHERE_RADIUS_FACTOR
    mat_s = rendering.MaterialRecord(); mat_s.shader="defaultLit"
    mat_s.base_color=(1.0,0,0,1.0)

    flann = o3d.geometry.KDTreeFlann(pcd)
    picking=False; selected=set(); sphere_names=set()

    def update_spheres():
        for n in sphere_names: scene.scene.remove_geometry(n)
        sphere_names.clear()
        pts = np.asarray(pcd.points)
        for idx in selected:
            sph=o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
            sph.compute_vertex_normals(); sph.translate(pts[idx])
            name=f"pick_{idx}"
            scene.scene.add_geometry(name,sph,mat_s)
            sphere_names.add(name)

    def on_key(ev):
        nonlocal picking
        if ev.type==gui.KeyEvent.Type.DOWN:
            if ev.key==gui.KeyName.F: picking=True; return True
            if ev.key==gui.KeyName.Q: window.close(); return True
        if ev.type==gui.KeyEvent.Type.UP and ev.key==gui.KeyName.F:
            picking=False; return True
        return False

    def on_mouse(ev):
        if not(picking and ev.type==gui.MouseEvent.Type.BUTTON_DOWN and ev.is_button_down(gui.MouseButton.LEFT)):
            return False
        x,y=ev.x-scene.frame.x,ev.y-scene.frame.y
        if not(0<=x<scene.frame.width and 0<=y<scene.frame.height):
            return False
        # depth callback
        def depth_cb(depth):
            d=np.asarray(depth)[y,x]
            if d>=1.0: return
            world=scene.scene.camera.unproject(x,y,d,scene.frame.width,scene.frame.height)
            _,ids,_=flann.search_knn_vector_3d(world,1)
            i0=ids[0]
            if i0 in selected: selected.remove(i0)
            else: selected.add(i0)
            update_spheres()
        scene.scene.scene.render_to_depth_image(depth_cb)
        return True

    window.set_on_key(on_key)
    scene.set_on_mouse(on_mouse)
    app.run()
    return list(selected)


def extract_roi_cloud(
    pcd: o3d.geometry.PointCloud,
    indices: list[int],
    expand: float=0.01
) -> tuple[o3d.geometry.PointCloud, o3d.geometry.OrientedBoundingBox]:
    """
    Из выбранных точек строит OBB (+ expand) и вырезает board_roi.
    Возвращает (board_roi, obb).
    """
    pts=np.asarray(pcd.points)[indices]
    obb=o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(pts))
    ext=obb.extent; new_ext=ext+np.array([expand]*3,dtype=ext.dtype)
    obb=o3d.geometry.OrientedBoundingBox(obb.center,obb.R,new_ext)
    return pcd.crop(obb), obb


def load_camera_params(path:str, cam_idx:int):
    """
    Считывает K,D из calib_cam_to_cam.txt и Tr_velo_to_cam из calib_velo_to_cam.txt
    """
    K,D,_,P=read_kitti_cam_calib(path,cam_idx)
    velo=Path(path).with_name("calib_velo_to_cam.txt")
    Rv,Tv=read_velo_to_cam(str(velo))
    Tr=np.hstack([Rv,Tv.reshape(3,1)])
    return K.astype(np.float32), D.astype(np.float32), Tr


def compute_board_frame(
    board_cloud: o3d.geometry.PointCloud
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    PCA-основанная локальная СК доски: origin, x_axis, y_axis, normal.
    """
    pts=np.asarray(board_cloud.points)
    origin=pts.mean(axis=0)
    cov=np.cov((pts-origin).T)
    vals,vecs=np.linalg.eigh(cov)
    # главные компоненты
    idx=np.argsort(vals)[::-1]
    x_axis=vecs[:,idx[0]]/np.linalg.norm(vecs[:,idx[0]])
    y_axis=vecs[:,idx[1]]/np.linalg.norm(vecs[:,idx[1]])
    normal=np.cross(x_axis,y_axis)
    normal/=np.linalg.norm(normal)
    # направить нормаль к камере (0,0,0)
    if np.dot(normal, -origin)<0:
        normal=-normal
    return origin, x_axis, y_axis, normal


def generate_object_points(
        origin: np.ndarray,
        x_axis: np.ndarray,
        y_axis: np.ndarray,
        pattern_size: tuple[int, int],
        square_size: float
) -> np.ndarray:
    """
    Генерирует идеальные 3D-координаты внутренних углов доски
    pattern_size = (cols, rows), square_size — длина клетки в м.
    Возвращает ndarray shape (N,3).
    """
    cols, rows = pattern_size
    pts3d = []
    # OpenCV corners идут в порядке row-major: сначала по j (rows), потом по i (cols)
    for j in range(rows):  # сначала строки (y-ось)
        for i in range(cols):  # внутри строки — столбцы (x-ось)
            pt = (origin
                  + i * square_size * x_axis
                  + j * square_size * y_axis)
            pts3d.append(pt)
    pts3d = np.array(pts3d, dtype=np.float32)
    # DEBUG: убедимся, что точек ровно cols*rows
    assert pts3d.shape[0] == cols * rows, \
        f"Expected {cols*rows} points, got {pts3d.shape[0]}"
    return pts3d

def compute_extrinsics(
    img_pts: np.ndarray,
    obj_pts: np.ndarray,
    K:       np.ndarray,
    D:       np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    EPnP + RANSAC, затем LM-уточнение.
    Возвращает (R, T, inlier_indices).
    """
    # приводим к нужному виду
    obj = obj_pts.reshape(-1,1,3)
    img = img_pts.reshape(-1,1,2)

    # 1) EPnP + RANSAC
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints   = obj,
        imagePoints    = img,
        cameraMatrix   = K,
        distCoeffs     = D,
        flags          = cv2.SOLVEPNP_EPNP,
        reprojectionError = 2.0,
        confidence        = 0.99,
        iterationsCount   = 100
    )
    if not success:
        raise RuntimeError("[ROI_SELECTOR] EPnP+RANSAC не сошёлся")

    inlier_idx = inliers.flatten().tolist()

    # 2) LM-уточнение по найденным inliers
    obj_in = obj_pts[inlier_idx].reshape(-1,1,3)
    img_in = img_pts[inlier_idx].reshape(-1,1,2)
    rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
        objectPoints = obj_in,
        imagePoints  = img_in,
        cameraMatrix = K,
        distCoeffs   = D,
        rvec         = rvec,
        tvec         = tvec
    )

    R, _ = cv2.Rodrigues(rvec_ref)
    T     = tvec_ref.flatten()
    return R, T, inlier_idx