# calibration/roi_selector/image_corners.py

import cv2
import numpy as np


def detect_image_corners(image: np.ndarray, pattern_size=(7, 5)) -> np.ndarray:
    """
    Находит углы шахматки на изображении и уточняет их субпиксельно.
    Возвращает массив N×2 с координатами (x, y) углов.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise RuntimeError(f"Не найдена шахматка {pattern_size}")
    # Уточняем позиции субпиксельно:
    cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    )
    return corners.reshape(-1, 2).astype(np.float32)


def adjust_corners_interactively(
    img: np.ndarray,
    corners: np.ndarray,
    pattern: tuple[int, int]
) -> np.ndarray:
    """
    Запускает окно, где рисунок (img) и прорисованные найденные углы шахматки.
    Пользователь может перетаскивать ближайший угол мышью, нажать ENTER/SPACE
    для подтверждения. Возвращает откорректированный массив углов.
    """
    win_name = "Adjust corners (drag & drop, Enter)"
    radius = 10  # радиус поиска ближайшего угла в пикселях

    dragging = False
    sel_idx = -1  # индекс выбранного угла для перетаскивания

    def on_mouse(event, x, y, flags, _):
        nonlocal dragging, sel_idx, corners

        if event == cv2.EVENT_LBUTTONDOWN:
            # если нажали ЛКМ — смотрим, не попали ли в область (radius) какого-нибудь угла
            for i, (cx, cy) in enumerate(corners):
                if (x - cx) ** 2 + (y - cy) ** 2 < radius**2:
                    sel_idx = i
                    dragging = True
                    break
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            # двигаем выбранный угол
            corners[sel_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            # отпустили ЛКМ — закончили перетаскивание
            dragging = False

    cv2.namedWindow(win_name)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        vis = img.copy()
        # Рисуем текущие углы
        cv2.drawChessboardCorners(vis, pattern, corners.reshape(-1, 1, 2), True)
        cv2.imshow(win_name, vis)

        key = cv2.waitKey(20)
        if key in (10, 13):  # Enter или Space
            break

    cv2.destroyWindow(win_name)
    return corners
