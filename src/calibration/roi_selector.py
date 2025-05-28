import numpy as np
import cv2
import open3d as o3d
from pathlib import Path
from utils.calib import read_kitti_cam_calib, read_velo_to_cam

SPHERE_RADIUS_FACTOR = 0.001

def detect_image_corners(image: np.ndarray, pattern_size=(7,5)) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
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
    p = Path(path)
    ext = p.suffix.lower()
    if ext in ('.pcd', '.ply'):
        pcd = o3d.io.read_point_cloud(str(p))
    elif ext == '.bin':
        data = np.fromfile(str(p), dtype=np.float32)
        pts = data.reshape(-1,4)[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    elif ext == '.txt':
        data = np.loadtxt(str(p), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        pts = data[:, :3]
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    else:
        raise ValueError(f"[ROI_SELECTOR] Неподдерживаемый формат: {ext}")
    if pcd.is_empty():
        raise ValueError(f"[ROI_SELECTOR] Пустое облако: {path}")
    return pcd


def detect_board_plane(
    pcd: o3d.geometry.PointCloud,
    pattern_size=(7,5),
    square_size=0.10,
    dist_thresh=0.005,
    visualize: bool = False
) -> tuple[list[float], o3d.geometry.PointCloud]:
    """
    Сегментирует плоскость и из её inliers выделяет единственный кластер доски.
    visualize=True отрисует board_cloud поверх серого pcd.
    """
    # 1) общая RANSAC-плоскость
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=dist_thresh,
        ransac_n=3,
        num_iterations=1000
    )
    inlier_cloud = pcd.select_by_index(inliers)

    # 2) кластеризация inlier_cloud
    labels = np.array(
        inlier_cloud.cluster_dbscan(eps=0.02, min_points=50, print_progress=False)
    )
    max_label = labels.max()
    rows, cols = pattern_size
    expected_dims = sorted([ (cols-1)*square_size, (rows-1)*square_size ])
    tol = max(expected_dims) * 0.2

    board_cloud = inlier_cloud  # по умолчанию
    for lbl in range(max_label + 1):
        idx = np.where(labels==lbl)[0]
        cluster = inlier_cloud.select_by_index(idx)
        obb = cluster.get_oriented_bounding_box()
        dims = sorted(obb.extent[:2])
        if abs(dims[0] - expected_dims[0]) < tol and abs(dims[1] - expected_dims[1]) < tol:
            board_cloud = cluster
            break

    # 3) визуализация для проверки
    if visualize:
        pcd.paint_uniform_color([0.7,0.7,0.7])
        board_cloud.paint_uniform_color([1.0,0.0,0.0])
        o3d.visualization.draw_geometries(
            [pcd, board_cloud],
            window_name="Gray: весь LiDAR, Red: предполагаемая доска"
        )

    return plane_model, board_cloud


def generate_board_corners_3d(
    plane_cloud: o3d.geometry.PointCloud,
    pattern_size=(7,5),
    square_size=0.10
) -> np.ndarray:
    pts = np.asarray(plane_cloud.points)
    mean = pts.mean(axis=0)
    cov = np.cov(pts - mean, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axes = eigvecs[:, np.argsort(-eigvals)[:2]]
    x_axis = axes[:,0] / np.linalg.norm(axes[:,0])
    y_axis = axes[:,1] / np.linalg.norm(axes[:,1])
    origin = mean

    rows, cols = pattern_size
    corners3d = []
    for i in range(rows):
        for j in range(cols):
            pt = origin + j*square_size*x_axis + i*square_size*y_axis
            corners3d.append(pt)
    return np.array(corners3d, dtype=np.float32)


def load_camera_params(calib_cam_path: str, cam_idx: int):
    K, D, R_rect, P_rect = read_kitti_cam_calib(calib_cam_path, cam_idx)
    velo_path = Path(calib_cam_path).with_name('calib_velo_to_cam.txt')
    Rv2c, Tv2c = read_velo_to_cam(str(velo_path))
    Tr = np.hstack([Rv2c, Tv2c.reshape(3,1)])
    return K, D, R_rect, P_rect, Tr


def compute_extrinsics(
    img_pts: np.ndarray,
    obj_pts: np.ndarray,
    K: np.ndarray,
    D: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    obj = obj_pts.reshape(-1,1,3)
    img = img_pts.reshape(-1,1,2)
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj, imagePoints=img,
        cameraMatrix=K, distCoeffs=D,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=2.0, confidence=0.99, iterationsCount=100
    )
    if not success:
        raise RuntimeError("[ROI_SELECTOR] EPnP + RANSAC не сошёлся")

    idx = inliers.flatten().tolist()
    obj_in = obj_pts[idx].reshape(-1,1,3)
    img_in = img_pts[idx].reshape(-1,1,2)
    rvec_ref, tvec_ref = cv2.solvePnPRefineLM(
        objectPoints=obj_in, imagePoints=img_in,
        cameraMatrix=K, distCoeffs=D,
        rvec=rvec, tvec=tvec
    )

    R, _ = cv2.Rodrigues(rvec_ref)
    T = tvec_ref.flatten()
    return R, T, idx
