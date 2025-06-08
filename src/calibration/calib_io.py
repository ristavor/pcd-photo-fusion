import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from numpy.typing import NDArray

from utils.calib import read_kitti_cam_calib


def create_ideal_camera_params(image_size: Tuple[int, int]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Creates ideal camera parameters (K and D) based on image size.
    
    Parameters:
        image_size (Tuple[int, int]): (width, height) of the image
        
    Returns:
        K (NDArray[float64], shape (3,3)): Ideal camera matrix
        D (NDArray[float64], shape (5,)): Zero distortion coefficients
    """
    width, height = image_size
    
    # Create ideal camera matrix
    # Principal point at image center
    cx = width / 2.0
    cy = height / 2.0
    
    # Focal length based on image size (typical for KITTI)
    fx = width * 0.8  # Typical value for KITTI cameras
    fy = fx  # Square pixels
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Zero distortion coefficients
    D = np.zeros(5, dtype=np.float64)
    
    return K, D


def load_camera_params(
    calib_cam_path: Optional[str],
    cam_idx: int,
    image_size: Optional[Tuple[int, int]] = None
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Загружает параметры камеры из файлов калибровки KITTI.
    Если calib_cam_path не указан, создает идеальные параметры камеры.

    Читаемые файлы:
      - calib_cam_to_cam.txt: содержит intrinsics (K), коэффициенты дисторсии (D),
        а также R_rect и P_rect (они здесь не используются).

    Параметры:
      calib_cam_path (Optional[str]): путь к файлу calib_cam_to_cam.txt. Если None,
        создаются идеальные параметры камеры.
      cam_idx (int): индекс камеры (например, 0, 1, …). Используется для выбора
        параметров K_cam_idx, D_cam_idx и т. д. внутри calib_cam_to_cam.txt.
      image_size (Optional[Tuple[int, int]]): (width, height) изображения. Требуется
        только если calib_cam_path не указан.

    Возвращает:
      K (NDArray[float64], shape (3,3)): матрица внутренних параметров камеры.
      D (NDArray[float64], shape (5,)): коэффициенты дисторсии (radial, tangential).

    Исключения:
      FileNotFoundError: если calib_cam_path не существует.
      KeyError: если в calib_cam_to_cam.txt отсутствуют необходимые записи для заданного cam_idx.
      ValueError: если формат файлов калибровки неверен или если image_size не указан при calib_cam_path=None.
    """
    if calib_cam_path is None:
        if image_size is None:
            raise ValueError("image_size must be provided when calib_cam_path is None")
        return create_ideal_camera_params(image_size)

    calib_path = Path(calib_cam_path)
    if not calib_path.is_file():
        raise FileNotFoundError(f"Файл калибровки камеры не найден: {calib_cam_path}")

    # Читаем K, D, R_rect, P_rect (R_rect и P_rect здесь игнорируются).
    try:
        K, D, _, _ = read_kitti_cam_calib(calib_cam_path, cam_idx)
    except KeyError as e:
        raise KeyError(f"Не удалось найти ключ {e.args[0]} в {calib_cam_path}") from None
    except Exception as e:
        raise ValueError(f"Ошибка при чтении {calib_cam_path}: {e}") from None

    # Приводим к float64 для совместимости с OpenCV
    return K.astype(np.float64), D.astype(np.float64)
