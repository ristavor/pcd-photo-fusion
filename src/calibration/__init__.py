"""
Пакет calibration: утилиты для калибровки камеры и LiDAR.

Подпакеты и модули:
  - image_corners: поиск и корректировка углов шахматки на изображении.
  - pcd_roi: загрузка и интерактивная разметка ROI в облаке точек.
  - board_geometry: вычисление локальной системы координат шахматной доски
    и генерация/уточнение 3D‐углов доски.
  - calib_io: чтение параметров камеры и LiDAR→Cam из файлов KITTI.
  - viz_utils: визуализация (проекция LiDAR на изображение, отображение R/T, и т. д.).

Из этого пакета удобно импортировать основные функции:
    from calibration import (
        detect_image_corners,
        adjust_corners_interactively,
        load_point_cloud,
        select_pointcloud_roi,
        extract_roi_cloud,
        compute_board_frame,
        generate_object_points,
        refine_3d_corners,
        load_camera_params,
        compute_axes_transform,
        reproject_and_show,
        draw_overlay,
        make_overlay_image
    )
"""

# Экспорт функций из подпакета image_corners
from .image_corners import detect_image_corners, adjust_corners_interactively

# Экспорт функций из подпакета pcd_roi
from .pcd_roi import load_point_cloud, select_pointcloud_roi, extract_roi_cloud

# Экспорт функций из подпакета board_geometry
from .board_geometry import compute_board_frame, generate_object_points, refine_3d_corners

# Экспорт функций из подпакета calib_io
from .calib_io import load_camera_params, compute_axes_transform

# Экспорт функций из подпакета viz_utils
from .viz_utils import reproject_and_show, draw_overlay, make_overlay_image

__all__ = [
    # image_corners
    "detect_image_corners",
    "adjust_corners_interactively",
    # pcd_roi
    "load_point_cloud",
    "select_pointcloud_roi",
    "extract_roi_cloud",
    # board_geometry
    "compute_board_frame",
    "generate_object_points",
    "refine_3d_corners",
    # calib_io
    "load_camera_params",
    "compute_axes_transform",
    # viz_utils
    "reproject_and_show",
    "draw_overlay",
    "make_overlay_image",
]
