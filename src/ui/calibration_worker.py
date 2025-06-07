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
from colorizer.colorizer import Colorizer


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
    Интерактивная корректировка R/T:
      - W/S/A/D/Q/E — повороты
      - I/K, J/L, U/O — смещения
      - M — показать интерактивную раскраску (вращаемую Open3D)
      - ESC — завершить и вернуть rvec/tvec
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
        # обновляем оверлей с текущими R и T
        draw_overlay(
            all_lidar_points,
            rvec,
            tvec.reshape(3, 1),
            K, D,
            image,
            window_name="Overlay"
        )

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
            break
        # вращение R
        elif key == ord('w'): rvec[0] -= delta_ang
        elif key == ord('s'): rvec[0] += delta_ang
        elif key == ord('a'): rvec[1] -= delta_ang
        elif key == ord('d'): rvec[1] += delta_ang
        elif key == ord('q'): rvec[2] -= delta_ang
        elif key == ord('e'): rvec[2] += delta_ang
        # смещение T
        elif key == ord('i'): tvec[0] -= delta_t
        elif key == ord('k'): tvec[0] += delta_t
        elif key == ord('j'): tvec[1] -= delta_t
        elif key == ord('l'): tvec[1] += delta_t
        elif key == ord('u'): tvec[2] -= delta_t
        elif key == ord('o'): tvec[2] += delta_t
        # показать интерактивную раскраску
        elif key == ord('m'):
            # пересчёт матрицы R из вектора
            R_mat = cv2.Rodrigues(rvec)[0]
            colorizer = Colorizer(R_mat, tvec.flatten(), K)
            pcd_colored = colorizer.colorize(all_lidar_points, image)

            # создаём **блокирующее** окно Open3D,
            # в котором сразу можно вращать облако мышкой
            vis = o3d.visualization.Visualizer()
            vis.create_window("Test Color (M)", width=800, height=600)
            vis.add_geometry(pcd_colored)
            vis.run()                 # <-- блокирует до закрытия окна
            vis.destroy_window()      # <-- после закрытия вы вернётесь в цикл

    cv2.destroyWindow("Overlay")
    return rvec, tvec


class CalibrationWorker(QThread):
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
            # 1. загрузить изображение
            self.progress.emit("Загрузка изображения...")
            img = cv2.imread(self.image_path)
            if img is None:
                raise RuntimeError(f"Не удалось загрузить изображение: {self.image_path}")

            # 2. ROI на изображении
            self.progress.emit("Выберите ROI шахматки...")
            x, y, w, h = map(int, cv2.selectROI("Выбор ROI", img))
            cv2.destroyWindow("Выбор ROI")
            if w == 0 or h == 0:
                raise RuntimeError("ROI не выбран.")
            roi_img = img[y:y+h, x:x+w]

            # 3. идеальные K, D
            self.progress.emit("Вычисление параметров K и D...")
            K, D = load_camera_params(img.shape[:2])

            # 4. детекция 2D-углов
            self.progress.emit("Детекция углов...")
            corners2d = detect_image_corners(roi_img, self.pattern)
            corners2d += np.array([x, y], dtype=np.float32)

            # 5. правка углов
            self.progress.emit("Корректировка 2D-углов...")
            corners2d = adjust_corners_interactively(img, corners2d, self.pattern)

            # 6. загрузка point cloud
            self.progress.emit("Загрузка облака точек...")
            pcd = load_point_cloud(self.cloud_path)

            # 7. ROI в облаке
            self.progress.emit("Выбор ROI в 3D...")
            indices = select_pointcloud_roi(pcd)
            if not indices:
                raise RuntimeError("ROI в 3D не выбран.")

            # 8. извлечение доски
            self.progress.emit("Извлечение доски...")
            board_roi, _ = extract_roi_cloud(pcd, indices)

            # 9. система координат доски
            self.progress.emit("Вычисление системы координат доски...")
            origin, x_axis, y_axis, _ = compute_board_frame(board_roi)

            # 10. генерация и уточнение 3D-углов
            self.progress.emit("Генерация 3D-углов...")
            obj_pts = generate_object_points(origin, x_axis, y_axis, self.pattern, self.square_size)
            refined_pts = refine_3d_corners(obj_pts, board_roi)

            # 11. PnP-конфигурации
            self.progress.emit("Решение PnP для конфигураций...")
            candidates = debug_pnp_axes(
                corners2d, origin, x_axis, y_axis,
                self.pattern, self.square_size,
                K, D
            )
            if not candidates:
                raise RuntimeError("PnP не дал решений.")

            # 12. выбор конфигурации
            idx = 0
            n = len(candidates)
            cv2.namedWindow("ChooseConfig", cv2.WINDOW_NORMAL)
            while True:
                name, mre, r0, t0 = candidates[idx]
                overlay = make_overlay_image(
                    np.asarray(pcd.points),
                    r0.reshape(3,1), t0.reshape(3,1),
                    K, D, img
                )
                cv2.putText(overlay, name, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
                cv2.putText(overlay, f"MRE={mre:.2f}px", (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)
                cv2.imshow("ChooseConfig", overlay)
                k = cv2.waitKey(0) & 0xFF
                if k in (13,10):
                    cv2.destroyWindow("ChooseConfig")
                    break
                elif k == ord('a'):
                    idx = (idx - 1) % n
                elif k == ord('d'):
                    idx = (idx + 1) % n

            # 13. интерактивная доводка R/T
            self.progress.emit(f"Выбрана {candidates[idx][0]}, MRE={candidates[idx][1]:.2f}px")
            self.progress.emit("Доводка R/T...")
            rvec_ref, tvec_ref = interactive_refine_RT(
                np.asarray(pcd.points),
                candidates[idx][2],
                candidates[idx][3],
                K, D,
                img
            )

            # 14. готово
            self.progress.emit("Калибровка завершена.")
            self.finished.emit(rvec_ref, tvec_ref)

        except Exception as e:
            self.error.emit(str(e))
