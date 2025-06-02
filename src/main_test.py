#!/usr/bin/env python3

import argparse
import cv2
import numpy as np

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
    generate_object_points
)
from calibration.calib_io import (
    load_camera_params
)
from calibration.viz_utils import (
    draw_overlay,
    make_overlay_image
)


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Вычисляет угол между двумя матрицами вращения (в градусах).
    """
    R = R_est @ R_gt.T
    cos_val = (np.trace(R) - 1) / 2
    cos_val = np.clip(cos_val, -1.0, 1.0)
    return np.degrees(np.arccos(cos_val))


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
) -> list[tuple[str, float, float, np.ndarray, np.ndarray]]:
    """
    Возвращает список из возможных конфигураций PnP:
      [(name, err_deg, t_err, R_est, T_est), …]
    Пользователь затем выбирает нужный.
    """
    cols, rows = pattern
    specs: list[tuple[bool, int, int, str]] = []
    for swap in (False, True):
        for sx in (1, -1):
            for sy in (1, -1):
                name = f"{'swap,' if swap else ''}{'+' if sx>0 else '-'}x,{'+' if sy>0 else '-'}y"
                specs.append((swap, sx, sy, name))

    results: list[tuple[str, float, float, np.ndarray, np.ndarray]] = []
    for swap, sx, sy, name in specs:
        if swap:
            xa = y_axis * sx
            ya = x_axis * sy
        else:
            xa = x_axis * sx
            ya = y_axis * sy

        pts3d = generate_object_points(origin, xa, ya, pattern, square_size)
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=pts3d.reshape(-1, 1, 3),
            imagePoints=corners2d.reshape(-1, 1, 2),
            cameraMatrix=K,
            distCoeffs=D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        R_est = cv2.Rodrigues(rvec)[0]
        T_est = tvec.flatten()
        err_deg = rotation_error_deg(R_est, R_gt)
        t_err = np.linalg.norm(T_est - T_gt.flatten())
        results.append((name, err_deg, t_err, R_est.astype(np.float32), T_est.astype(np.float32)))

    return results


def interactive_refine_RT(
    all_lidar_points: np.ndarray,
    rvec_init: np.ndarray,
    tvec_init: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    image: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Интерактивная корректировка:
      - rvec можно крутить кнопками (W/S/A/D/Q/E) и мышью (перетаскивание ЛКМ).
      - tvec меняется кнопками (I/K, J/L, U/O).
    Возвращает (rvec, tvec) при нажатии ESC.
    """
    rvec = rvec_init.copy()
    tvec = tvec_init.copy()

    delta_ang = 0.01   # шаг вращения (≈0.57°)
    delta_t = 0.01     # шаг трансляции (≈1 см)
    mouse_sens = 0.005  # чувствительность для мыши

    dragging = False
    last_x = last_y = 0

    def on_mouse(event, x, y, flags, _):
        nonlocal dragging, last_x, last_y, rvec
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            last_x, last_y = x, y
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            dx = x - last_x
            dy = y - last_y
            rvec[1] += dx * mouse_sens  # yaw
            rvec[0] += dy * mouse_sens  # pitch
            last_x, last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False

    cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Overlay", on_mouse)

    while True:
        draw_overlay(all_lidar_points, rvec, tvec.reshape(3, 1), K, D, image, window_name="Overlay")

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break

        # Вращение R
        if   key == ord('w'): rvec[0] -= delta_ang
        elif key == ord('s'): rvec[0] += delta_ang
        elif key == ord('a'): rvec[1] -= delta_ang
        elif key == ord('d'): rvec[1] += delta_ang
        elif key == ord('q'): rvec[2] -= delta_ang
        elif key == ord('e'): rvec[2] += delta_ang

        # Сдвиг T
        elif key == ord('i'): tvec[0] -= delta_t
        elif key == ord('k'): tvec[0] += delta_t
        elif key == ord('j'): tvec[1] -= delta_t
        elif key == ord('l'): tvec[1] += delta_t
        elif key == ord('u'): tvec[2] -= delta_t
        elif key == ord('o'): tvec[2] += delta_t

    cv2.destroyWindow("Overlay")
    return rvec, tvec


def main():
    parser = argparse.ArgumentParser(
        description="Диагностика PnP + интерактивная корректировка R и T"
    )
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="четное число путей: img0 pcd0 img1 pcd1 …"
    )
    parser.add_argument(
        "--calib", required=True,
        help="путь к calib_cam_to_cam.txt (рядом calib_velo_to_cam.txt)"
    )
    parser.add_argument(
        "--camidx", type=int, default=0,
        help="индекс камеры в calib_cam_to_cam.txt (обычно 0)"
    )
    parser.add_argument(
        "--pattern", nargs=2, type=int, default=[7, 5],
        help="cols rows внутренних углов шахматки"
    )
    parser.add_argument(
        "--square_size", type=float, default=0.10,
        help="размер клетки шахматки (в метрах)"
    )
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        raise ValueError("--pairs должно содержать чётное число путей")

    # Читаем intrinsics и эталонные extrinsics
    K, D, R_gt, T_gt = load_camera_params(args.calib, args.camidx)

    for i in range(0, len(args.pairs), 2):
        img_path = args.pairs[i]
        pcd_path = args.pairs[i + 1]
        print(f"\n=== Обработка кадра {i//2}: {img_path} + {pcd_path} ===")

        # --- 2D: загрузка и ROI изображения ---
        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            continue

        x, y, w, h = map(int, cv2.selectROI("ROI", img))
        cv2.destroyWindow("ROI")
        if w == 0 or h == 0:
            print("ROI не выбран, пропускаем этот кадр.")
            continue

        roi_img = img[y:y + h, x:x + w]
        try:
            corners2d = detect_image_corners(roi_img, tuple(args.pattern))
        except RuntimeError as e:
            print(f"Ошибка поиска углов: {e}")
            continue

        corners2d += np.array([x, y], dtype=np.float32)
        corners2d = adjust_corners_interactively(img, corners2d, tuple(args.pattern))

        # --- 3D: загрузка LiDAR-облака и ROI ---
        try:
            pcd = load_point_cloud(pcd_path)
        except (ValueError, RuntimeError) as e:
            print(f"Ошибка загрузки point cloud: {e}")
            continue

        idxs = select_pointcloud_roi(pcd)
        if not idxs:
            print("ROI точек не выбран, пропускаем этот кадр.")
            continue

        board_roi, _ = extract_roi_cloud(pcd, idxs)
        origin, x_axis, y_axis, _ = compute_board_frame(board_roi)

        # --- 4: PnP: собираем все варианты, без автоматического выбора ---
        candidates = debug_pnp_axes(
            corners2d, origin, x_axis, y_axis,
            tuple(args.pattern), args.square_size,
            K, D, R_gt, T_gt
        )
        if not candidates:
            print("PNP не дал ни одного решения, пропускаем этот кадр.")
            continue

        # --- 5: интерактивный выбор конфигурации ---
        idx = 0
        n = len(candidates)
        cv2.namedWindow("ChooseConfig", cv2.WINDOW_NORMAL)

        while True:
            name, err_deg, err_t, R_est, T_est = candidates[idx]
            rvec, _ = cv2.Rodrigues(R_est)
            tvec = T_est.reshape(3, 1)

            overlay_img = make_overlay_image(np.asarray(pcd.points), rvec, tvec, K, D, img)
            cv2.putText(overlay_img, name, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay_img, f"rot_err={err_deg:.2f}°, t_err={err_t:.3f}m",
                        (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("ChooseConfig", overlay_img)
            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10):  # Enter
                cv2.destroyWindow("ChooseConfig")
                R_best_mat, T_best = R_est, T_est
                break
            elif key == ord('a'):
                idx = (idx - 1) % n
            elif key == ord('d'):
                idx = (idx + 1) % n

        print(f">> Выбран вариант: R=\n{R_best_mat}\nT={T_best}\n")

        # --- 6: отображаем первоначальное наложение с найденными R и T ---
        all_lidar = np.asarray(pcd.points)
        rvec_best, _ = cv2.Rodrigues(R_best_mat)
        tvec_best = T_best.reshape(3, 1)
        draw_overlay(all_lidar, rvec_best, tvec_best, K, D, img, window_name="Overlay")
        cv2.waitKey(1)

        # --- 7: интерактивная корректировка R и T ---
        print(
            ">> Входим в интерактивный режим корректировки.\n"
            "   Вращение R: W/S (X), A/D (Y), Q/E (Z)\n"
            "   Трансляция T: I/K (X), J/L (Y), U/O (Z)\n"
            "   ESC → завершить"
        )
        rvec_refined, tvec_refined = interactive_refine_RT(
            all_lidar_points=all_lidar,
            rvec_init=rvec_best,
            tvec_init=tvec_best,
            K=K, D=D,
            image=img
        )

        R_refined_mat = cv2.Rodrigues(rvec_refined)[0]
        T_refined = tvec_refined.flatten()
        print(
            f">> Итоговый rvec_refined:\n{rvec_refined.flatten()}\n"
            f"   Соответствующая R_refined:\n{R_refined_mat}\n"
            f"   Итоговый tvec_refined: {T_refined}\n"
        )

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
