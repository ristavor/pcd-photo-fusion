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
from calibration.viz_utils import reproject_and_show, draw_overlay


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
    выводит reprojection‐ошибку по углу и погрешность T, сортирует.
    Возвращает список результатов [(name, err_rot, err_t, R_est, T_est), …].
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
        # 1) задаём ориентированные оси
        if swap:
            xa = y_axis * sx
            ya = x_axis * sy
        else:
            xa = x_axis * sx
            ya = y_axis * sy

        # 2) генерируем 3D‐координаты углов шахматки
        pts3d = generate_object_points(origin, xa, ya, pattern, square_size)
        # при желании можно уточнить 3D‐координаты углов:
        # pts3d = refine_3d_corners(pts3d, board_cloud, k=10)

        # 3) SolvePnP (ITERATIVE)
        ok, rvec, tvec = cv2.solvePnP(
            objectPoints=pts3d.reshape(-1, 1, 3),
            imagePoints=corners2d.reshape(-1, 1, 2),
            cameraMatrix=K,
            distCoeffs=D,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue

        # 4) Уточняем только rvec, T_gt фиксирован
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

    # Сортируем по углу (меньше лучше), затем по ΔT (что тут =0 для всех)
    results.sort(key=lambda x: (x[1], x[2]))

    print("\n--- Варианты (название, угол_ошибки°, ΔT_норма) ---")
    for name, e_deg, te, rmat, tvec in results:
        print(f"{name:12s}  err_rot={e_deg:6.2f}°  err_t={te:5.3f}m\n{rmat}\n{tvec}\n")
    best = results[0]
    print(f">> Лучший вариант: {best[0]}  → err_rot={best[1]:.2f}°, err_t={best[2]:.3f}m\n")
    return results  # список из кортежей (см. выше)


def interactive_refine_R(
    all_lidar_points: np.ndarray,
    rvec_init: np.ndarray,
    tvec_fixed: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    image: np.ndarray
) -> np.ndarray:
    """
    Запускает интерактивный цикл, позволяющий «клавишами» (W/S, A/D, Q/E)
    слегка корректировать rvec_init вдоль осей X, Y, Z, сохраняя tvec_fixed.
    Возвращает итоговый rvec (3×1) после нажатия ESC.

    Маппинг клавиш:
      W/S → вращение вокруг X (уменьш/увелич угла)
      A/D → вращение вокруг Y
      Q/E → вращение вокруг Z
      ESC → выход, вернуть итоговый rvec
    """
    # 1) Начальная точка
    rvec = rvec_init.copy()

    # 2) Шаг изменения (в радианах)
    delta = 0.01  # ≈0.57°

    # 3) Открываем окно Overlay
    cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)

    while True:
        # Каждый раз рисуем наложение "облако → картинка" с текущим rvec
        draw_overlay(all_lidar_points, rvec, tvec_fixed, K, D, image, window_name="Overlay")

        # Ждём нажатие клавиши
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            break

        # Поправки по клавишам:
        if key == ord('w'):     # ↑ поворот вокруг X, «вверх»
            rvec[0] -= delta
        elif key == ord('s'):   # ↓ поворот вокруг X, «вниз»
            rvec[0] += delta
        elif key == ord('a'):   # ← поворот вокруг Y, «влево»
            rvec[1] -= delta
        elif key == ord('d'):   # → поворот вокруг Y, «вправо»
            rvec[1] += delta
        elif key == ord('q'):   # Q поворот вокруг Z (против часовой)
            rvec[2] -= delta
        elif key == ord('e'):   # E поворот вокруг Z (по часовой)
            rvec[2] += delta
        # Иначе: любая другая клавиша — игнорируем, ждем дальше

        # При следующей итерации цикла снова перерисуем «Overlay» с новым rvec

    cv2.destroyWindow("Overlay")
    return rvec


def main():
    parser = argparse.ArgumentParser(
        description="Диагностика PnP (фиксируем T, ищем R) + интерактивная «подкрутка» R"
    )
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="четное число путей: img0 pcd0 img1 pcd1 ..."
    )
    parser.add_argument(
        "--calib", required=True,
        help="calib_cam_to_cam.txt (рядом calib_velo_to_cam.txt)"
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
        help="размер клетки шахматки (м)"
    )
    args = parser.parse_args()

    if len(args.pairs) % 2 != 0:
        raise ValueError("--pairs должен содержать чётное число путей")

    # 1) Читаем intrinsics + эталонный extrinsics (R_gt, T_gt)
    K, D, R_gt, T_gt = load_camera_params(args.calib, args.camidx)
    # 2) Матрица R_axes (LiDAR→Camera axes‐only), нам пригодится, если надо
    R_axes = compute_axes_transform()

    for i in range(0, len(args.pairs), 2):
        img_path = args.pairs[i]
        pcd_path = args.pairs[i + 1]
        print(f"\n=== Кадр {i // 2}: {img_path} + {pcd_path} ===")

        # --- 2D: ROI + поиск/коррекция углов шахматки ---
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Не найдено изображение {img_path}")
        x, y, w, h = map(int, cv2.selectROI("ROI", img))
        cv2.destroyWindow("ROI")
        roi_img = img[y : y + h, x : x + w]

        corners2d = detect_image_corners(roi_img, tuple(args.pattern))
        # Переносим координаты найденных углов на полное изображение
        corners2d += np.array([x, y], dtype=np.float32)
        corners2d = adjust_corners_interactively(img, corners2d, tuple(args.pattern))

        # --- 3D: загружаем LiDAR-облако, отмечаем ROI, вычисляем локальную СК доски ---
        pcd = load_point_cloud(pcd_path)
        idxs = select_pointcloud_roi(pcd)
        board_roi, _ = extract_roi_cloud(pcd, idxs)
        origin, x_axis, y_axis, normal = compute_board_frame(board_roi)

        # --- 4) Перебираем «зеркальные» варианты R --}}
        results = debug_pnp_axes(
            corners2d, origin, x_axis, y_axis,
            tuple(args.pattern), args.square_size,
            K, D, R_gt, T_gt
        )

        # --- 5) Автоматически взяли «best» R из списка: ---
        best_name, _, _, R_best_mat, T_best = results[0]
        print(f">> Лучший автоматический вариант: '{best_name}' → R_best_mat:\n{R_best_mat}\nT_fixed={T_best}\n")

        # --- 6) Показываем Overlay (облако → картинка) с этим R_best: ---
        all_lidar = np.asarray(pcd.points)  # (N×3) — все точки из выбранного pcd
        rvec_best, _ = cv2.Rodrigues(R_best_mat)  # (3×1) вектор Rodrigues
        print(">> Отображаем первоначальный Overlay для R_best...")
        draw_overlay(all_lidar, rvec_best, T_best.reshape(3, 1), K, D, img, window_name="Overlay")
        cv2.waitKey(1)  # небольшой debounce, чтобы окно успело появиться

        # --- 7) Запускаем интерактивный режим, чтобы пользователь «подкрутил» R вручную: ---
        print(">> Войдите в интерактивный режим корректировки R. \n"
              "   Клавиши: W/S (X‐ось), A/D (Y‐ось), Q/E (Z‐ось), ESC → закончить.")
        rvec_refined = interactive_refine_R(
            all_lidar_points=all_lidar,
            rvec_init=rvec_best,
            tvec_fixed=T_best.reshape(3, 1),
            K=K, D=D,
            image=img
        )

        # --- 8) После ESC получаем итоговый rvec_refined: ---
        R_refined_mat = cv2.Rodrigues(rvec_refined)[0]
        print(f">> Итоговый rvec_refined (Rodrigues):\n{rvec_refined.flatten()}\n"
              f"   и соответствующая R_refined:\n{R_refined_mat}\n"
              f"   T оставили прежним: {T_best}\n")

        # --- 9) (Опционально) здесь можете сохранить R_refined_mat и T_best куда-нибудь в файл ---
        # Например:
        # np.savetxt("R_final.txt", R_refined_mat, fmt="%.6f")
        # np.savetxt("T_final.txt", T_best, fmt="%.6f")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
