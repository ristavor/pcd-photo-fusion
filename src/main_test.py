#!/usr/bin/env python3
# main_test.py

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
from calibration.calib_io import load_camera_params
from calibration.viz_utils import (
    draw_overlay,
    make_overlay_image
)


def debug_pnp_axes(
    corners2d: np.ndarray,
    origin: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    pattern: tuple[int, int],
    square_size: float,
    K: np.ndarray,
    D: np.ndarray
) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    """
    Для каждого варианта (swap, ±x, ±y) решает PnP и считает
    среднюю ошибку проекции (MRE) в пикселях.

    Возвращает список кортежей:
      (name, mre_px, rvec (3,), tvec (3,))
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
        xa = (y_axis if swap else x_axis) * sx
        ya = (x_axis if swap else y_axis) * sy
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

        # проецируем обратно и считаем MRE
        proj, _ = cv2.projectPoints(pts3d.reshape(-1, 1, 3), rvec, tvec, K, D)
        proj2d = proj.reshape(-1, 2)
        mre = float(np.mean(np.linalg.norm(proj2d - corners2d, axis=1)))

        results.append((name, mre, rvec.flatten(), tvec.flatten()))

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
      - R (rvec) вращается кнопками W/S/A/D/Q/E и мышью.
      - T (tvec) сдвигается кнопками I/K, J/L, U/O.
    ESC завершает и возвращает (rvec, tvec).
    """
    rvec = rvec_init.copy()
    tvec = tvec_init.copy()

    delta_ang = 0.01   # ≈0.57°
    delta_t   = 0.01   # ≈1 см
    mouse_sens = 0.005

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
        description="Диагностика PnP + интерактивная корректировка R и T (идеальная камера)"
    )
    parser.add_argument(
        "--pairs", nargs="+", required=True,
        help="четное число путей: img0 pcd0 img1 pcd1 …"
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

    pattern = tuple(args.pattern)
    sq_size = args.square_size

    for i in range(0, len(args.pairs), 2):
        img_path = args.pairs[i]
        pcd_path = args.pairs[i + 1]
        print(f"\n=== Кадр {i//2}: {img_path} + {pcd_path} ===")

        # 1) Загрузка и K, D
        img = cv2.imread(img_path)
        if img is None:
            print(f"Не удалось загрузить изображение: {img_path}")
            continue
        K, D = load_camera_params(img.shape[:2])

        # 2) ROI на изображении
        x, y, w, h = map(int, cv2.selectROI("ROI", img))
        cv2.destroyWindow("ROI")
        if w == 0 or h == 0:
            print("ROI не выбран, пропускаем кадр.")
            continue
        roi_img = img[y:y+h, x:x+w]

        # 3) Детекция и уточнение углов
        try:
            corners2d = detect_image_corners(roi_img, pattern)
        except RuntimeError as e:
            print(f"Ошибка поиска углов: {e}")
            continue
        corners2d += np.array([x, y], dtype=np.float32)
        corners2d = adjust_corners_interactively(img, corners2d, pattern)

        # 4) Загрузка и ROI LiDAR
        try:
            pcd = load_point_cloud(pcd_path)
        except (ValueError, RuntimeError) as e:
            print(f"Ошибка загрузки point cloud: {e}")
            continue
        idxs = select_pointcloud_roi(pcd)
        if not idxs:
            print("ROI точек не выбран, пропускаем кадр.")
            continue
        board_roi, _ = extract_roi_cloud(pcd, idxs)

        # 5) Вычисление системы координат доски
        origin, x_axis, y_axis, _ = compute_board_frame(board_roi)

        # 6) PnP: варианты и MRE
        candidates = debug_pnp_axes(
            corners2d, origin, x_axis, y_axis,
            pattern, sq_size, K, D
        )
        if not candidates:
            print("PNP не дал ни одного решения, пропускаем кадр.")
            continue

        # 7) Интерактивный выбор по MRE
        idx = 0
        n = len(candidates)
        cv2.namedWindow("ChooseConfig", cv2.WINDOW_NORMAL)
        while True:
            name, mre, rvec, tvec = candidates[idx]
            overlay = make_overlay_image(
                np.asarray(pcd.points),
                rvec.reshape(3,1),
                tvec.reshape(3,1),
                K, D, img
            )
            cv2.putText(overlay, name, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, f"MRE={mre:.2f}px", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imshow("ChooseConfig", overlay)

            key = cv2.waitKey(0) & 0xFF
            if key in (13, 10):  # Enter
                cv2.destroyWindow("ChooseConfig")
                break
            elif key == ord('a'):
                idx = (idx - 1) % n
            elif key == ord('d'):
                idx = (idx + 1) % n

        # 8) Получаем лучший вариант и показываем первоначальное наложение
        _, _, rvec_best, tvec_best = candidates[idx]
        draw_overlay(
            np.asarray(pcd.points),
            rvec_best.reshape(3,1),
            tvec_best.reshape(3,1),
            K, D, img,
            window_name="Overlay"
        )
        cv2.waitKey(1)

        # 9) Интерактивная корректировка
        print(
            ">> Интерактивный режим корректировки:\n"
            "   Вращение R: W/S (X), A/D (Y), Q/E (Z)\n"
            "   Трансляция T: I/K (X), J/L (Y), U/O (Z)\n"
            "   ESC → завершить"
        )
        rvec_refined, tvec_refined = interactive_refine_RT(
            all_lidar_points=np.asarray(pcd.points),
            rvec_init=rvec_best,
            tvec_init=tvec_best,
            K=K, D=D,
            image=img
        )

        # 10) Вывод итоговых R (9 элементов) и tvec
        R_refined = cv2.Rodrigues(rvec_refined)[0]  # 3×3
        R_flat = R_refined.flatten()               # 9 элементов
        t_flat = tvec_refined.flatten()            # 3 элемента
        print(f">> Итоговая R (flattened, 9 элем): {R_flat.tolist()}")
        print(f">> Итоговый t (3 элем): {t_flat.tolist()}\n")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
