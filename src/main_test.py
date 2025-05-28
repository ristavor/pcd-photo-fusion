#!/usr/bin/env python3
# File: src/main_test.py

import argparse
import cv2
import numpy as np
from calibration.roi_selector import detect_image_corners

def main():
    parser = argparse.ArgumentParser(
        description="Интерактивный ROI + проверка детекции углов шахматной доски"
    )
    parser.add_argument(
        "-i", "--image", required=True,
        help="Путь к исходному изображению (PNG/JPG)"
    )
    parser.add_argument(
        "--pattern", nargs=2, type=int, default=[7, 5],
        help="Число внутренних углов доски: cols rows (по умолчанию 7 5)"
    )
    args = parser.parse_args()

    # 1) Загружаем изображение
    img = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {args.image}")

    pattern_size = (args.pattern[0], args.pattern[1])
    print(f">>> Подвиньте рамку вокруг нужной доски и нажмите ENTER/SPACE")

    # 2) Пользователь выбирает ROI на картинке
    cv2.namedWindow("Select ROI", cv2.WINDOW_NORMAL)
    roi = cv2.selectROI("Select ROI", img)
    cv2.destroyWindow("Select ROI")
    x, y, w, h = map(int, roi)
    if w == 0 or h == 0:
        print("ROI не выбран, выходим.")
        return

    # 3) Покажем, куда попал ROI
    img_roi = img[y:y+h, x:x+w]
    vis = img.copy()
    cv2.rectangle(vis, (x,y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imshow("Chosen ROI", vis)
    cv2.waitKey(500)
    cv2.destroyWindow("Chosen ROI")

    # 4) Детектируем углы внутри ROI
    print(f">>> Ищем {pattern_size[0]}×{pattern_size[1]} углов внутри ROI ...")
    try:
        corners_roi = detect_image_corners(img_roi, pattern_size)
    except Exception as e:
        print("Ошибка при детекции углов:", e)
        return

    # 5) Смещаем координаты углов в систему целого изображения
    corners2d = corners_roi + np.array([x, y], dtype=np.float32)
    # 5.1 — Показываем ROI с углами
    vis_roi = img_roi.copy()
    cv2.drawChessboardCorners(vis_roi, pattern_size,
                              corners_roi.reshape(-1, 1, 2), True)
    cv2.namedWindow("ROI with corners", cv2.WINDOW_NORMAL)
    cv2.imshow("ROI with corners", vis_roi)

    # 5.2 — Показываем полный кадр с ROI и смещёнными углами
    vis_full = img.copy()
    # зелёная рамка ROI
    cv2.rectangle(vis_full, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.drawChessboardCorners(vis_full, pattern_size,
                              corners2d.reshape(-1, 1, 2), True)
    cv2.namedWindow("Full image with corners", cv2.WINDOW_NORMAL)
    cv2.imshow("Full image with corners", vis_full)

    print("Нажмите любую клавишу, если кружки в ROI и на общем кадре совпадают по положению")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 6) Визуализируем результат на полном изображении
    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern_size,
                              corners2d.reshape(-1, 1, 2), True)
    window_name = "Detected Corners in ROI"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, vis)
    print("Нажмите любую клавишу, если все точки лежат на пересечениях клеток...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
