#!/usr/bin/env python3
# src/main.py

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from colorizer import Colorizer


def load_rt(rt_path: Path) -> (np.ndarray, np.ndarray):
    """
    Считывает R и T из JSON:
    {
      "R": [[...], [...], [...]],
      "T": [...]
    }
    Возвращает R (3×3 np.float64), T (3,) np.float64.
    """
    with open(rt_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    R = np.array(data['R'], dtype=np.float64)
    T = np.array(data['T'], dtype=np.float64)
    return R, T


def load_point_cloud(path: Path) -> np.ndarray:
    """
    Считывает облако точек и возвращает N×3 numpy-массив XYZ.
    Поддерживаются .pcd/.ply, .bin (float32×4), .txt (текст).
    """
    ext = path.suffix.lower()
    if ext in ('.pcd', '.ply'):
        pcd = o3d.io.read_point_cloud(str(path))
        pts = np.asarray(pcd.points, dtype=np.float64)
    elif ext == '.bin':
        arr = np.fromfile(str(path), dtype=np.float32).reshape(-1, 4)
        pts = arr[:, :3].astype(np.float64)
    elif ext == '.txt':
        arr = np.loadtxt(str(path), dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        pts = arr[:, :3]
    else:
        raise ValueError(f"Unsupported point cloud extension: {ext}")
    return pts


def build_ideal_K(image: np.ndarray) -> np.ndarray:
    """
    Строит «идеальную» матрицу K по размерам image:
      f = max(width, height)
      cx = width/2, cy = height/2
    """
    h, w = image.shape[:2]
    f = float(max(w, h))
    cx = w / 2.0
    cy = h / 2.0

    K = np.array([
        [f,   0.0, cx],
        [0.0, f,   cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return K


def colorize_and_visualize(
    image: np.ndarray,
    points: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    K: np.ndarray,
    window_name: str = "Colored PointCloud"
) -> o3d.visualization.Visualizer:
    """
    Окрашивает облако points по пикселям из image с помощью Colorizer(R,T,K).
    Возвращает готовый к отображению Visualizer.
    """
    colorizer = Colorizer(R, T, K)
    pcd_colored: o3d.geometry.PointCloud = colorizer.colorize(points, image)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, width=800, height=600)
    vis.add_geometry(pcd_colored)
    return vis


def main():
    parser = argparse.ArgumentParser(
        description="Окрасить своё облако точек по своему RT и изображению")
    parser.add_argument(
        "--rt", required=True, type=Path,
        help="JSON-файл с полями R (3×3) и T (3)")
    parser.add_argument(
        "--pcd", required=True, type=Path,
        help="Файл облака точек (.pcd/.ply/.bin/.txt)")
    parser.add_argument(
        "--img", required=True, type=Path,
        help="Файл изображения (любой, поддерживаемый OpenCV)")
    args = parser.parse_args()

    # 1) RT
    R, T = load_rt(args.rt)

    # 2) Загрузка изображения
    image = cv2.imread(str(args.img), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: не удалось загрузить изображение {args.img}")
        return

    # 3) Загрузка облака точек
    pts = load_point_cloud(args.pcd)

    # 4) Идеальная K
    K = build_ideal_K(image)

    # 5) Цветная визуализация
    vis = colorize_and_visualize(
        image=image,
        points=pts,
        R=R,
        T=T,
        K=K,
        window_name="Colored PointCloud"
    )

    # 6) Цикл отображения (ESC для выхода)
    try:
        while True:
            vis.poll_events()
            vis.update_renderer()
            if cv2.waitKey(1) == 27:
                break
    except KeyboardInterrupt:
        pass
    finally:
        vis.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
