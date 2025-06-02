import cv2
import numpy as np
from typing import Tuple
from numpy.typing import NDArray


def detect_image_corners(
    image: NDArray[np.uint8],
    pattern_size: Tuple[int, int] = (7, 5)
) -> NDArray[np.float32]:
    """
    Находит внутренние углы шахматной доски на изображении и уточняет их субпиксельно.

    Алгоритм:
      1. Переводит изображение в оттенки серого.
      2. Вызывает cv2.findChessboardCorners для поиска внутренних углов
         размером pattern_size = (cols, rows).
      3. Если chessboard не найден, бросает RuntimeError.
      4. При успешном обнаружении применяет cv2.cornerSubPix для уточнения
         позиций углов с субпиксельной точностью.

    Параметры:
      image (NDArray[uint8], shape (H,W,3) или (H,W)): BGR‐изображение.
      pattern_size (Tuple[int,int]): кортеж (cols, rows) внутренних углов
        шахматки (например, (7,5) означает 7 углов по ширине и 5 по высоте).

    Возвращает:
      corners (NDArray[float32], shape (N,2)): координаты N = cols*rows углов
        в формате [[x1, y1], [x2, y2], …], упорядоченные построчно.

    Исключения:
      RuntimeError: если шахматка не найдена.
      ValueError: если изображение некорректного формата.
    """
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError("На входе должно быть корректное NumPy‐изображение.")
    # Если цветное, переводим в оттенки серого
    if image.ndim == 3 and image.shape[2] in (3, 4):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError("Изображение должно быть 2D или 3D с 3 или 4 каналами.")

    # Ищем углы шахматки
    found, corners = cv2.findChessboardCorners(
        gray,
        pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        raise RuntimeError(f"Шахматка размером {pattern_size} не найдена на изображении.")

    # Уточняем позиции субпиксельно
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1)
    cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=criteria
    )
    # corners имеет shape (N,1,2) → приводим к (N,2)
    corners = corners.reshape(-1, 2).astype(np.float32)
    return corners


def adjust_corners_interactively(
    img: NDArray[np.uint8],
    corners: NDArray[np.float32],
    pattern: Tuple[int, int],
    radius: int = 10
) -> NDArray[np.float32]:
    """
    Запускает интерактивное окно, где пользователь может «подтянуть» найденные углы
    шахматки, перетаскивая их мышью, и подтвердить результат клавишей Enter или Space.

    Логика:
      1. Рисует исходное изображение с отмеченными углами.
      2. При клике ЛКМ проверяет, попал ли курсор в круг радиуса radius
         вокруг ближайшего угла; если да, начинает «драг» этого угла.
      3. Пока ЛКМ зажат и мышь двигается, обновляет координаты выбранного угла.
      4. При отпускании ЛКМ завершает драг.
      5. Когда пользователь нажимает Enter (код 13) или Space (код 32), функция
         закрывает окно и возвращает скорректированные углы.

    Параметры:
      img (NDArray[uint8], shape (H,W,3)): BGR‐изображение полного кадра.
      corners (NDArray[float32], shape (N,2)): найденные углы шахматки
        в координатах полного кадра. Массив изменяется «на месте».
      pattern (Tuple[int,int]): (cols, rows) внутренних углов chessboard.
      radius (int): радиус в пикселях для выбора ближайшего угла при клике
        (по умолчанию 10).

    Возвращает:
      corners (NDArray[float32], shape (N,2)): скорректированные углы.

    Исключения:
      ValueError: если входные параметры некорректны.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Для корректировки углов нужно передать корректное изображение.")
    if corners.ndim != 2 or corners.shape[1] != 2:
        raise ValueError("Corners должен быть массивом формы (N,2).")
    if not (isinstance(pattern, tuple) and len(pattern) == 2):
        raise ValueError("Pattern должен быть кортежем (cols, rows).")
    if radius <= 0:
        raise ValueError("Радиус должен быть положительным целым числом.")

    win_name = "Adjust corners (drag & drop, Enter/Space)"
    dragging = False
    sel_idx = -1  # индекс выбранного угла

    def on_mouse(event, x, y, flags, _):
        nonlocal dragging, sel_idx, corners
        if event == cv2.EVENT_LBUTTONDOWN:
            # Проверяем ближайший угол
            for i, (cx, cy) in enumerate(corners):
                if (x - cx) ** 2 + (y - cy) ** 2 < radius**2:
                    sel_idx = i
                    dragging = True
                    break
        elif event == cv2.EVENT_MOUSEMOVE and dragging and sel_idx >= 0:
            # Обновляем координаты выбранного угла
            corners[sel_idx] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            sel_idx = -1

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)

    while True:
        vis = img.copy()
        # Рисуем найденные или скорректированные углы
        cv2.drawChessboardCorners(
            vis,
            pattern,
            corners.reshape(-1, 1, 2),
            True
        )
        cv2.imshow(win_name, vis)
        key = cv2.waitKey(20) & 0xFF
        # Enter (13) или Space (32) → подтверждение
        if key in (13, 32):
            break

    cv2.destroyWindow(win_name)
    return corners
