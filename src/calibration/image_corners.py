# calibration/image_corners.py

import cv2
import numpy as np
from typing import Tuple
from numpy.typing import NDArray

__all__ = [
    "detect_image_corners",
    "adjust_corners_interactively",
]


def detect_image_corners(
        image: NDArray[np.uint8],
        pattern_size: Tuple[int, int]
) -> NDArray[np.float32]:
    """
    Находит внутренние углы шахматной доски с адаптивным предобработкой и
    мульти-масштабной детекцией (SB и классическим).

    :param image: BGR или grayscale NumPy-изображение
    :param pattern_size: (cols, rows) внутренних углов шахматки
    :return: corners (N×2) float32 — координаты углов
    :raises RuntimeError: если углы не найдены
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("На входе должно быть NumPy‐изображение.")
    # 1) Перевод в оттенки серого
    if image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image.copy()
    else:
        raise ValueError("Изображение должно быть 2D или BGR(3/4).")

    # 2) Оценка резкости через дисперсию лапласиана
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())
    blur_threshold = 1000.0
    is_sharp = blur_var > blur_threshold

    # 3) Предобработка только если размыто
    if not is_sharp:
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.equalizeHist(gray)

    # 4) Функция SB-детектора
    def sb_detect(img: NDArray[np.uint8]):
        flags_sb = (
                cv2.CALIB_CB_NORMALIZE_IMAGE
                | cv2.CALIB_CB_EXHAUSTIVE
                | cv2.CALIB_CB_ACCURACY
                | cv2.CALIB_CB_LARGER
        )
        return cv2.findChessboardCornersSB(img, pattern_size, flags=flags_sb)

    # 5) Попытка SB-детекции на оригинале, затем на 2×
    found, corners = sb_detect(gray)
    if not found:
        gray_up = cv2.pyrUp(gray)
        found_up, corners_up = sb_detect(gray_up)
        if found_up:
            corners = corners_up * 0.5
            found = True

    # 6) Фоллбэк к классическому детектору
    if not found:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        raise RuntimeError(f"Шахматка размером {pattern_size} не найдена.")

    # 7) Субпиксельное уточнение с динамическими параметрами
    if is_sharp:
        win_size = (11, 11)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    else:
        win_size = (21, 21)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 0.005)

    cv2.cornerSubPix(
        gray,
        corners,
        winSize=win_size,
        zeroZone=(-1, -1),
        criteria=criteria
    )

    return corners.reshape(-1, 2).astype(np.float32)


def adjust_corners_interactively(
        img: NDArray[np.uint8],
        corners: NDArray[np.float32],
        pattern: Tuple[int, int],
        radius: int = 10
) -> NDArray[np.float32]:
    """
    Интерактивная корректировка найденных углов шахматки (drag & drop).

    :param img: BGR-изображение для отрисовки
    :param corners: найденные углы (N×2)
    :param pattern: кортеж (cols, rows) внутренних углов
    :param radius: радиус выбора угла (в пикселях)
    :return: скорректированные углы (N×2)
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Нужно передать корректное изображение.")
    if corners.ndim != 2 or corners.shape[1] != 2:
        raise ValueError("Corners должен быть формы (N,2).")
    if not (isinstance(pattern, tuple) and len(pattern) == 2):
        raise ValueError("Pattern должен быть кортежем (cols, rows).")
    if radius <= 0:
        raise ValueError("Radius должен быть положительным.")

    win_name = "Adjust corners (drag & drop, Enter/Space)"
    dragging = False
    sel_idx = -1

    def on_mouse(event, x, y, flags, _):
        nonlocal dragging, sel_idx, corners
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, (cx, cy) in enumerate(corners):
                if (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2:
                    sel_idx = i
                    dragging = True
                    break
        elif event == cv2.EVENT_MOUSEMOVE and dragging and sel_idx >= 0:
            corners[sel_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            sel_idx = -1

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        vis = img.copy()
        cv2.drawChessboardCorners(vis, pattern, corners.reshape(-1, 1, 2), True)
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 32):  # Enter или Space
            break

    cv2.destroyWindow(win_name)
    return corners
