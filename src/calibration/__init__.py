# calibration/roi_selector/__init__.py

# Этот файл делает пакет roi_selector. Здесь можно импортировать наиболее
# часто используемые функции, чтобы при импорте из roi_selector их было
# удобно брать напрямую, но мы можем и оставить его пустым,
# снабдив явными submodule-путями в main_test.py.

__all__ = [
    # image_corners
    "detect_image_corners", "adjust_corners_interactively",
    # pcd_roi
    "load_point_cloud", "select_pointcloud_roi", "extract_roi_cloud",
    # board_geometry
    "compute_board_frame", "generate_object_points", "refine_3d_corners",
    # calib_io
    "load_camera_params", "compute_axes_transform",
    # viz_utils
    "reproject_and_show"
]

# По желанию можно пробросить через __init__ функции из подмодулей:
from .image_corners import detect_image_corners, adjust_corners_interactively
from .pcd_roi import load_point_cloud, select_pointcloud_roi, extract_roi_cloud
from .board_geometry import compute_board_frame, generate_object_points, refine_3d_corners
from .calib_io import load_camera_params, compute_axes_transform
from .viz_utils import reproject_and_show
