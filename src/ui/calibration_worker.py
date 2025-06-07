# src/ui/calibration_worker.py

import cv2
import numpy as np
import open3d as o3d
from PySide6.QtCore import QThread, Signal

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
from calibration.calib_io import load_camera_params
from calibration.viz_utils import draw_overlay, make_overlay_image


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
    Возвращает список (name, mre_px, rvec (3,), tvec (3,)).
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
      - вращение R: W/S/A/D/Q/E + мышь
      - сдвиг T: I/K, J/L, U/O
    ESC завершает и возвращает (rvec, tvec).
    """
    rvec = rvec_init.copy()
    tvec = tvec_init.copy()

    delta_ang = 0.01
    delta_t   = 0.01
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
            rvec[1] += dx * mouse_sens
            rvec[0] += dy * mouse_sens
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
        # R: W/S/A/D/Q/E
        if   key == ord('w'): rvec[0] -= delta_ang
        elif key == ord('s'): rvec[0] += delta_ang
        elif key == ord('a'): rvec[1] -= delta_ang
        elif key == ord('d'): rvec[1] += delta_ang
        elif key == ord('q'): rvec[2] -= delta_ang
        elif key == ord('e'): rvec[2] += delta_ang
        # T: I/K, J/L, U/O
        elif key == ord('i'): tvec[0] -= delta_t
        elif key == ord('k'): tvec[0] += delta_t
        elif key == ord('j'): tvec[1] -= delta_t
        elif key == ord('l'): tvec[1] += delta_t
        elif key == ord('u'): tvec[2] -= delta_t
        elif key == ord('o'): tvec[2] += delta_t

    cv2.destroyWindow("Overlay")
    return rvec, tvec


class CalibrationWorker(QThread):
    """
    Выполняет пайплайн калибровки в отдельном потоке:
      1. Выбор ROI изображения
      2. Детекция и коррекция 2D-углов
      3. Выбор ROI в 3D и реконструкция доски
      4. Интерактивный выбор конфигурации PnP
      5. Интерактивная донастройка R/T
    Сигналы:
      progress(str) — сообщения для лога/статуса
      error(str)    — при ошибках завершает работу
      finished(object, object) — rvec_refined, tvec_refined
    """
    progress = Signal(str)
    error    = Signal(str)
    finished = Signal(object, object)

    def __init__(
        self,
        square_size: float,
        cols: int,
        rows: int,
        image_path: str,
        cloud_path: str,
        parent=None
    ):
        super().__init__(parent)
        self.square_size = square_size
        self.pattern = (cols, rows)
        self.image_path = image_path
        self.cloud_path = cloud_path

    def run(self):
        try:
            # 1. Загрузка изображения
            self.progress.emit("Загрузка изображения...")
            img = cv2.imread(self.image_path)
            if img is None:
                raise RuntimeError(f"Не удалось загрузить изображение: {self.image_path}")

            # 2. Выбор области шахматки вручную
            self.progress.emit("Выберите область шахматки на изображении...")
            x, y, w, h = map(int, cv2.selectROI("Выбор ROI", img))
            cv2.destroyWindow("Выбор ROI")
            if w == 0 or h == 0:
                raise RuntimeError("ROI на изображении не выбран.")
            roi_img = img[y:y+h, x:x+w]

            # 3. Параметры камеры (идеальные K, D)
            self.progress.emit("Вычисление параметров K и D...")
            K, D = load_camera_params(img.shape[:2])

            # 4. Детекция углов в ROI
            self.progress.emit("Детекция углов шахматки в ROI...")
            corners2d = detect_image_corners(roi_img, self.pattern)
            corners2d += np.array([x, y], dtype=np.float32)

            # 5. Интерактивная корректировка 2D-углов
            self.progress.emit("Интерактивная корректировка внутренних углов...")
            corners2d = adjust_corners_interactively(img, corners2d, self.pattern)

            # 6. Загрузка point cloud
            self.progress.emit("Загрузка point cloud...")
            pcd = load_point_cloud(self.cloud_path)

            # 7. Выбор ROI в облаке точек
            self.progress.emit("Выбор ROI в облаке точек...")
            indices = select_pointcloud_roi(pcd)
            if not indices:
                raise RuntimeError("ROI в облаке точек не выбран.")

            # 8. Извлечение облака доски
            self.progress.emit("Извлечение облака доски...")
            board_roi, _ = extract_roi_cloud(pcd, indices)

            # 9. Вычисление системы координат доски
            self.progress.emit("Вычисление системы координат доски...")
            origin, x_axis, y_axis, _ = compute_board_frame(board_roi)

            # 10. Генерация и уточнение 3D-углов
            self.progress.emit("Генерация и уточнение 3D-углов...")
            obj_pts = generate_object_points(origin, x_axis, y_axis, self.pattern, self.square_size)
            refined_pts = refine_3d_corners(obj_pts, board_roi)

            # 11. Решение PnP для разных конфигураций
            self.progress.emit("Решение задач PnP для разных конфигураций...")
            candidates = debug_pnp_axes(
                corners2d, origin, x_axis, y_axis,
                self.pattern, self.square_size,
                K, D
            )
            if not candidates:
                raise RuntimeError("PnP не дал ни одного решения.")

            # 12. Интерактивный выбор конфигурации
            idx = 0
            n = len(candidates)
            cv2.namedWindow("ChooseConfig", cv2.WINDOW_NORMAL)
            while True:
                name, mre, rvec0, tvec0 = candidates[idx]
                overlay = make_overlay_image(
                    np.asarray(pcd.points),
                    rvec0.reshape(3,1),
                    tvec0.reshape(3,1),
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

            # 13. Интерактивная корректировка R/T
            self.progress.emit(f"Выбрана конфигурация {candidates[idx][0]}, MRE={candidates[idx][1]:.2f}px")
            self.progress.emit("Интерактивная корректировка R/T...")
            rvec_refined, tvec_refined = interactive_refine_RT(
                all_lidar_points=np.asarray(pcd.points),
                rvec_init=candidates[idx][2],
                tvec_init=candidates[idx][3],
                K=K, D=D,
                image=img
            )

            # 14. Завершение
            self.progress.emit("Калибровка завершена.")
            self.finished.emit(rvec_refined, tvec_refined)

        except Exception as e:
            self.error.emit(str(e))
