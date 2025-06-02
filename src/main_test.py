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
from calibration.viz_utils import reproject_and_show, draw_overlay, make_overlay_image


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Угол между двумя матрицами вращения в градусах.
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
    Возвращает список из 8 вариантов [(name, err_deg, t_err, R_est, T_est), …],
    без автоматической сортировки и выбора.
    Пользователь будет сам выбирать подходящий конфиг.
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
        # (при желании: pts3d = refine_3d_corners(pts3d, board_cloud, k=10))

        # 3) SolvePnP (ITERATIVE) с оценкой и rvec, tvec
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

        results.append((name, err_deg, t_err, R_est, T_est))

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

    # Параметры «шагов» при вращении с клавиатуры:
    delta_ang = 0.01    # ≈0.57° за нажатие
    delta_t   = 0.01    # 1 см за нажатие

    # Коэффициент «чувствительности» для мышиного вращения:
    mouse_sensitivity = 0.005  # можно регулировать

    # Переменные для mouse-callback-а:
    dragging = False
    last_x, last_y = 0, 0

    # Функция-обработчик мыши:
    def on_mouse(event, x, y, flags, _):
        nonlocal dragging, last_x, last_y, rvec

        if event == cv2.EVENT_LBUTTONDOWN:
            # Начали тянуть мышь
            dragging = True
            last_x, last_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            # Пока тащат, вычисляем смещение от последней точки
            dx = x - last_x
            dy = y - last_y

            # Обновляем rvec:
            # будем считать, что dx → вращение вокруг Y, а dy → вокруг X
            rvec[1] += dx * mouse_sensitivity  # «yaw»
            rvec[0] += dy * mouse_sensitivity  # «pitch»

            # «Сдвигаем» базовую точку для следующей итерации
            last_x, last_y = x, y

        elif event == cv2.EVENT_LBUTTONUP:
            # Конец перетаскивания
            dragging = False

    # Подвязываем обработчик к окну «Overlay»
    cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Overlay", on_mouse)

    while True:
        # 1) Рисуем наложение (draw_overlay внутри себя делает imshow+waitKey(1))
        draw_overlay(all_lidar_points, rvec, tvec.reshape(3, 1), K, D, image, window_name="Overlay")

        # 2) Ждём 30 мс для клавиатуры
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break

        # 3) Обработка нажатий клавиш для R
        if   key == ord('w'): rvec[0] -= delta_ang
        elif key == ord('s'): rvec[0] += delta_ang
        elif key == ord('a'): rvec[1] -= delta_ang
        elif key == ord('d'): rvec[1] += delta_ang
        elif key == ord('q'): rvec[2] -= delta_ang
        elif key == ord('e'): rvec[2] += delta_ang

        # 4) Обработка клавиш для T
        elif key == ord('i'): tvec[0] -= delta_t
        elif key == ord('k'): tvec[0] += delta_t
        elif key == ord('j'): tvec[1] -= delta_t
        elif key == ord('l'): tvec[1] += delta_t
        elif key == ord('u'): tvec[2] -= delta_t
        elif key == ord('o'): tvec[2] += delta_t

        # Если key == −1 (никакой клавиши) или другие клавиши, то цикл просто перерисует.

    cv2.destroyWindow("Overlay")
    return rvec, tvec





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

        candidates = debug_pnp_axes(
            corners2d, origin, x_axis, y_axis,
            tuple(args.pattern), args.square_size,
            K, D, R_gt, T_gt
        )
        # Теперь – интерактивный выбор:
        R_best_mat, T_best = interactive_choose_configuration(
            all_lidar=np.asarray(pcd.points),
            image=img,
            K=K, D=D,
            candidates=candidates
        )
        # Можно вывести в консоль, что выбрали:
        print(f">> Выбранный вариант: R=\n{R_best_mat}\nT={T_best}\n")

        # --- 6) Показываем Overlay (облако → картинка) с этим R_best и T_best: ---
        all_lidar = np.asarray(pcd.points)  # (N×3) — все точки из выбранного pcd
        rvec_best, _ = cv2.Rodrigues(R_best_mat)  # (3×1) Rodrigues-ветор для лучшего варианта
        tvec_best = T_best.reshape(3, 1)  # (3×1) вектор трансляции из PnP

        print(">> Отображаем первоначальный Overlay для R_best и T_best...")
        draw_overlay(all_lidar, rvec_best, tvec_best, K, D, img, window_name="Overlay")
        cv2.waitKey(1)  # небольшой debounce, чтобы окно успело появиться

        # --- 7) Запускаем интерактивный режим для корректировки R и T: ---
        print(">> Входим в интерактивный режим корректировки R и T. \n"
              "   Клавиши для вращения R: W/S (X‐ось), A/D (Y‐ось), Q/E (Z‐ось);\n"
              "   Клавиши для трансляции T: I/K (X), J/L (Y), U/O (Z);\n"
              "   ESC → закончить и сохранить текущее состояние.")
        rvec_refined, tvec_refined = interactive_refine_RT(
            all_lidar_points=all_lidar,
            rvec_init=rvec_best,
            tvec_init=tvec_best,
            K=K, D=D,
            image=img
        )

        # --- 8) После ESC получаем итоговые rvec_refined и tvec_refined: ---
        R_refined_mat = cv2.Rodrigues(rvec_refined)[0]  # окончательная матрица вращения
        T_refined = tvec_refined.flatten()  # окончательный вектор трансляции

        print(f">> Итоговый rvec_refined (Rodrigues):\n{rvec_refined.flatten()}\n"
              f"   и соответствующая R_refined:\n{R_refined_mat}\n"
              f"   Итоговый tvec_refined: {T_refined}\n")

        # --- 9) (Опционально) здесь можете сохранить R_refined_mat и T_best куда-нибудь в файл ---
        # Например:
        # np.savetxt("R_final.txt", R_refined_mat, fmt="%.6f")
        # np.savetxt("T_final.txt", T_best, fmt="%.6f")

    cv2.destroyAllWindows()


def interactive_choose_configuration(
    all_lidar: np.ndarray,
    image: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    candidates: list[tuple[str, float, float, np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    idx = 0
    n = len(candidates)
    cv2.namedWindow("ChooseConfig", cv2.WINDOW_NORMAL)

    while True:
        name, err_deg, err_t, R_est, T_est = candidates[idx]
        rvec, _ = cv2.Rodrigues(R_est)
        tvec = T_est.reshape(3, 1)

        # 1) Получаем кадр с точками (без текста)
        overlay_img = make_overlay_image(all_lidar, rvec, tvec, K, D, image)

        # 2) Рисуем текст name и ошибки поверх overlay_img:
        #    (желательно жёлтый с чёрной обводкой — по желанию)
        #    Но для проверки оставим белый, как раньше:
        cv2.putText(overlay_img, name,
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay_img, f"rot_err={err_deg:.2f}°, t_err={err_t:.3f}m",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 2, cv2.LINE_AA)

        # 3) Показываем итоговый кадр с точками + текстом
        cv2.imshow("ChooseConfig", overlay_img)

        key = cv2.waitKey(0) & 0xFF
        if key in (13, 10):  # Enter (13 на Windows, 10 на Linux/Mac)
            cv2.destroyWindow("ChooseConfig")
            return R_est, T_est

        # вместо стрелок «←/→» — используем 'a' / 'd'
        elif key == ord('a'):  # 'a' — переход «влево»
            idx = (idx - 1) % n
        elif key == ord('d'):  # 'd' — переход «вправо»
            idx = (idx + 1) % n
        # любая другая клавиша — ничего не делаем, остаёмся на тек. idx


if __name__ == "__main__":
    main()
