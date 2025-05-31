# main_test.py

#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import open3d as o3d
from scipy.optimize import least_squares

from calibration.image_corners import (
    detect_image_corners,
    adjust_corners_interactively
)
from calibration.pcd_roi import (
    load_point_cloud,
    select_pointcloud_roi,
    extract_roi_cloud
)
from calibration.board_geometry import (
    compute_board_frame,
    generate_object_points,
    refine_3d_corners
)
from calibration.calib_io import (
    load_camera_params,
    compute_axes_transform
)
from calibration.viz_utils import reproject_and_show


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Угол между двумя матрицами вращения в градусах.
    """
    R = R_est @ R_gt.T
    cos_val = (np.trace(R) - 1) / 2
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


def refine_r_only(
    pts3d: np.ndarray,
    pts2d: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    rvec0: np.ndarray,
    T_gt: np.ndarray
) -> np.ndarray:
    """
    LM-оптимизация только rvec, при фиксированном T_gt.
    Возвращает уточнённый rvec (3×1).
    """
    def residuals(r):
        r = r.reshape(3, 1)
        proj, _ = cv2.projectPoints(
            pts3d.reshape(-1, 1, 3),
            r,
            T_gt.reshape(3, 1),
            K,
            D
        )
        return (proj.reshape(-1, 2) - pts2d).reshape(-1)

    x0 = rvec0.flatten()
    sol = least_squares(residuals, x0, method='lm')
    return sol.x.reshape(3, 1)


def debug_pnp_axes(
    corners2d: np.ndarray,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    pattern: tuple[int, int],
    square_size: float,
    K: np.ndarray,
    D: np.ndarray,
    R_gt: np.ndarray,
    T_gt: np.ndarray
):
    """
    Перебирает 8 возможных конфигураций ориентации доски (±x, ±y, swap),
    выводит для каждой reproj-ошибку по углу и погрешность T (хоть T фиксирован),
    сортирует по наименьшей rot-погрешности, печатает все варианты.
    """
    cols, rows = pattern
    specs = []
    for swap in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                name = f"{'swap,' if swap else ''}{'+' if sx>0 else '-'}x,{'+' if sy>0 else '-'}y"
                specs.append((swap, sx, sy, name))

    results = []
    for swap, sx, sy, name in specs:
        # задаём ориентированные оси
        if swap:
            xa = y_axis * sx
            ya = x_axis * sy
        else:
            xa = x_axis * sx
            ya = y_axis * sy

        # генерируем 3D-точки углов шахматки
        pts3d = generate_object_points(origin, xa, ya, pattern, square_size)
        # при желании можно уточнить 3D-координаты углов:
        # pts3d = refine_3d_corners(pts3d, board_cloud, k=10)

        # PnP (iterative) без догадки (rvec0=0,tvec0=0)
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=pts3d.reshape(-1, 1, 3),
            imagePoints=corners2d.reshape(-1, 1, 2),
            cameraMatrix=K,
            distCoeffs=D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        # уточняем только rvec, T_gt фиксировано
        rvec_ref = refine_r_only(
            pts3d, corners2d, K, D,
            rvec.reshape(3, 1),
            T_gt
        )
        R_est = cv2.Rodrigues(rvec_ref)[0]
        T_est = T_gt.flatten()

        err_deg = rotation_error_deg(R_est, R_gt)
        t_err = np.linalg.norm(T_est - T_gt.flatten())

        results.append((name, err_deg, t_err, R_est, T_est))

    # сортируем по углу (меньше лучше), затем по T‐ошибке (что всегда =0 здесь)
    results.sort(key=lambda x: (x[1], x[2]))

    print("\n--- Варианты (название, угол_ошибки°, ΔT_норма) ---")
    for name, e_deg, te, rmat, tvec in results:
        print(f"{name:12s}  err_rot={e_deg:6.2f}°  err_t={te:5.3f}m\n{rmat}\n{tvec}\n")
    best = results[0]
    print(f">> Лучший вариант: {best[0]}  → err_rot={best[1]:.2f}°, err_t={best[2]:.3f}m\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Диагностика PnP (фиксируем T, ищем R) — 'mirror ambiguity'"
    )
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="четное число путей: img0 pcd0 img1 pcd1 ..."
    )
    parser.add_argument(
        "--calib", required=True,
        help="calib_cam_to_cam.txt (рядом находится calib_velo_to_cam.txt)"
    )
    parser.add_argument(
        "--camidx", type=int, default=0,
        help="индекс камеры в calib_cam_to_cam.txt (обычно 0)"
    )
    parser.add_argument(
        "--pattern", nargs=2, type=int, default=[7, 5],
        help="cols rows внутренней сетки шахматки"
    )
    parser.add_argument(
        "--square_size", type=float, default=0.10,
        help="размер квадрата шахматки (м)"
    )
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        raise ValueError("--pairs должен содержать чётное число путей")

    # 1) Читаем intrinsics + эталонный extrinsics (R_gt, T_gt)
    K, D, R_gt, T_gt = load_camera_params(args.calib, args.camidx)
    # 2) Матрица осей LiDAR→CAM (без трансляции)
    R_axes = compute_axes_transform()

    for i in range(0, len(args.pairs), 2):
        img_path = args.pairs[i]
        pcd_path = args.pairs[i + 1]
        print(f"\n=== Кадр {i // 2}: {img_path} + {pcd_path} ===")

        # --- 2D: выбираем ROI + находим / корректируем углы шахматки ---
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Не найдено изображение {img_path}")
        x, y, w, h = map(int, cv2.selectROI("ROI", img))
        cv2.destroyWindow("ROI")
        roi_img = img[y : y + h, x : x + w]

        corners2d = detect_image_corners(roi_img, tuple(args.pattern))
        # переносим координаты на полное изображение
        corners2d += np.array([x, y], dtype=np.float32)
        corners2d = adjust_corners_interactively(img, corners2d, tuple(args.pattern))

        # --- 3D: загружаем LiDAR-облако, выделяем ROI (точки шахматки), считаем локальную СК ---
        pcd = load_point_cloud(pcd_path)
        idxs = select_pointcloud_roi(pcd)
        board_roi, _ = extract_roi_cloud(pcd, idxs)
        origin, x_axis, y_axis, normal = compute_board_frame(board_roi)

        # --- 4) Диагностика «зеркальных» вариантов R ---
        results = debug_pnp_axes(
            corners2d, origin, x_axis, y_axis,
            tuple(args.pattern), args.square_size,
            K, D, R_gt, T_gt
        )

        # --- 5) (Опционально) показ Overlay-результата для лучшего понимания ---
        # Берём лучший вариант:
        best_name, _, _, R_best, T_best = results[0]
        print(f">> Отображаем Overlay для варианта '{best_name}'")
        # Для Overlay нужны все LiDAR-точки из полного кадра:
        all_lidar = np.asarray(pcd.points)  # (N×3)
        rvec_best, _ = cv2.Rodrigues(R_best)
        reproject_and_show(all_lidar, rvec_best, T_best.reshape(3, 1), K, D, img, window_name="Overlay")

        # --- 6) Пользователь может вручную корректировать R (например, через trackbar),
        #     но это выходит за рамки данного скрипта. Здесь показан автоматический результат. ---

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
