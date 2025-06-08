# main.py
# !/usr/bin/env python3
import sys
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from colorizer import read_velo_to_cam, Colorizer
from rectifier import ImageRectifier
from synchronizer import Synchronizer


def load_velo(path: Path) -> np.ndarray:
    """.bin через fromfile, .txt через loadtxt. Возвращает N×3 XYZ."""
    if path.suffix == '.bin':
        pts = np.fromfile(str(path), dtype=np.float32)
        pts = pts.reshape(-1, 4)
        return pts[:, :3]
    elif path.suffix == '.txt':
        pts = np.loadtxt(str(path), dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        return pts[:, :3]
    else:
        raise ValueError(f"Unsupported extension: {path.suffix}")


def color_and_show(img: np.ndarray, pts: np.ndarray, window_name: str, colorizer: Colorizer):
    pcd = colorizer.colorize(pts, img)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name, 800, 600)
    vis.add_geometry(pcd)
    return vis


def main():
    base = Path(__file__).resolve().parent.parent
    raw_root = base / "data" / "2011_09_28_drive_0034_extract"
    sync_root = base / "data" / "2011_09_28_drive_0034_sync"
    cam_folder = "image_02"

    print("[main.py] base:", base)
    print("[main.py] raw_root:", raw_root)
    print("[main.py] sync_root:", sync_root)
    print("[main.py] cam_folder:", cam_folder)

    # 1) Полная синхронизация raw
    syncer = Synchronizer(raw_root=raw_root, cam_folder=cam_folder)
    matches = syncer.match_pairs()
    if not matches:
        print("Нет подходящих пар по timestamps.")
        sys.exit(1)

    # 2) Вывод таблицы RAW
    print("RAW\nНОМЕР - ФОТО - ОБЛАКО")
    for idx, (c, v) in enumerate(matches):
        print(f"{idx:>4} - {c:>4} - {v:>4}")

    # 3) Ввод от пользователя
    raw_choice, sync_choice = map(int, input("\nВВЕДИТЕ ДВА ИНДЕКСА (raw sync): ").split())

    # Проверка
    if not (0 <= raw_choice < len(matches)) or sync_choice < 0:
        print("Индексы вне диапазона.")
        sys.exit(1)

    # 4) Подготовка colorizer
    calib_dir = base / "data" / "2011_09_28_calib"
    print("[main.py] calib_dir:", calib_dir)
    velo_calib_path = calib_dir / "calib_velo_to_cam.txt"
    cam_calib_path = calib_dir / "calib_cam_to_cam.txt"
    print("[main.py] velo_calib_path:", velo_calib_path)
    print("[main.py] cam_calib_path:", cam_calib_path)
    R, T = read_velo_to_cam(velo_calib_path)
    print("[main.py] R (velo_to_cam):\n", R)
    print("[main.py] T (velo_to_cam):", T)

    # 5) Обработка raw-пары
    raw_cam_idx, raw_velo_idx = matches[raw_choice]
    print(f"[main.py] raw_cam_idx: {raw_cam_idx}, raw_velo_idx: {raw_velo_idx}")
    # load & rectify image
    img_path = raw_root / cam_folder / "data" / f"{raw_cam_idx:010d}.png"
    print("[main.py] img_path:", img_path)
    img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    print("[main.py] img_raw.shape:", img_raw.shape if img_raw is not None else None)
    cam_idx = int(cam_folder.split('_')[-1])
    print("[main.py] cam_idx:", cam_idx)
    rectifier = ImageRectifier(calib_cam_path=cam_calib_path, cam_idx=cam_idx)
    print("[main.py] rectifier.P_new (P_rect[:3, :3]):\n", rectifier.P_new)
    img_raw_rect = rectifier.rectify(img_raw)
    print("[main.py] img_raw_rect.shape:", img_raw_rect.shape)
    # load velo
    velo_path = raw_root / "velodyne_points" / "data" / f"{raw_velo_idx:010d}.txt"
    print("[main.py] velo_path:", velo_path)
    velo_raw = load_velo(velo_path)
    print("[main.py] velo_raw.shape:", velo_raw.shape)

    # 6) Обработка sync-пары
    sync_cam_idx = sync_choice
    sync_velo_idx = sync_choice
    img_sync_path = sync_root / cam_folder / "data" / f"{sync_cam_idx:010d}.png"
    print("[main.py] img_sync_path:", img_sync_path)
    img_sync = cv2.imread(str(img_sync_path), cv2.IMREAD_UNCHANGED)
    print("[main.py] img_sync.shape:", img_sync.shape if img_sync is not None else None)
    velo_sync_path = sync_root / "velodyne_points" / "data" / f"{sync_velo_idx:010d}.bin"
    print("[main.py] velo_sync_path:", velo_sync_path)
    velo_sync = load_velo(velo_sync_path)
    print("[main.py] velo_sync.shape:", velo_sync.shape)

    # 7) Colorizer (используем одну и ту же матрицу K для raw и sync)
    colorizer = Colorizer(R, T, rectifier.P_new)
    print("[main.py] Colorizer K (rectifier.P_new):\n", rectifier.P_new)

    # 8) Визуализация
    print("[main.py] --- Визуализация RAW ---")
    vis_raw = color_and_show(img_raw_rect, np.asarray(velo_raw), f"RAW idx={raw_choice} ({raw_cam_idx}-{raw_velo_idx})",
                             colorizer)
    print("[main.py] --- Визуализация SYNC ---")
    vis_sync = color_and_show(img_sync, np.asarray(velo_sync), f"SYNC idx={sync_choice}", colorizer)

    try:
        while True:
            cv2.imshow("RAW image", img_raw_rect)
            cv2.imshow("SYNC image", img_sync)
            if cv2.waitKey(1) == 27:
                break
            vis_raw.poll_events();
            vis_raw.update_renderer()
            vis_sync.poll_events();
            vis_sync.update_renderer()
    except KeyboardInterrupt:
        pass
    finally:
        vis_raw.destroy_window()
        vis_sync.destroy_window()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
