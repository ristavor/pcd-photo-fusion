import numpy as np
from pathlib import Path
from typing import Tuple
from numpy.typing import NDArray

from utils.calib import read_kitti_cam_calib, read_velo_to_cam


def load_camera_params(
    calib_cam_path: str,
    cam_idx: int
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Загружает параметры камеры и LiDAR→Cam (extrinsics) из файлов калибровки KITTI.

    Читаемые файлы:
      - calib_cam_to_cam.txt: содержит intrinsics (K), коэффициенты дисторсии (D),
        а также R_rect и P_rect (они здесь не используются).
      - calib_velo_to_cam.txt: содержит R (3×3) и T (3,) для преобразования
        LiDAR→Camera.

    Параметры:
      calib_cam_path (str): путь к файлу calib_cam_to_cam.txt.
      cam_idx (int): индекс камеры (например, 0, 1, …). Используется для выбора
        параметров K_cam_idx, D_cam_idx и т. д. внутри calib_cam_to_cam.txt.

    Возвращает:
      K (NDArray[float64], shape (3,3)): матрица внутренних параметров камеры.
      D (NDArray[float64], shape (5,)): коэффициенты дисторсии (radial, tangential).
      R_gt (NDArray[float64], shape (3,3)): эталонная матрица вращения LiDAR→Cam.
      T_gt (NDArray[float64], shape (3,)): эталонный вектор трансляции LiDAR→Cam.

    Исключения:
      FileNotFoundError: если calib_cam_path не существует или рядом нет calib_velo_to_cam.txt.
      KeyError: если в calib_cam_to_cam.txt отсутствуют необходимые записи для заданного cam_idx.
      ValueError: если формат файлов калибровки неверен.
    """
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

    # Проверяем наличие файла calib_velo_to_cam.txt в той же папке
    velo_path = calib_path.with_name("calib_velo_to_cam.txt")
    if not velo_path.is_file():
        raise FileNotFoundError(f"Файл LiDAR→Cam не найден: {velo_path}")

    try:
        R_gt, T_gt = read_velo_to_cam(str(velo_path))
    except KeyError as e:
        raise KeyError(f"Не удалось найти ключ {e.args[0]} в {velo_path}") from None
    except Exception as e:
        raise ValueError(f"Ошибка при чтении {velo_path}: {e}") from None

    # Приводим к float64 для совместимости с OpenCV
    return K.astype(np.float64), D.astype(np.float64), R_gt.astype(np.float64), T_gt.astype(np.float64)


def compute_axes_transform() -> NDArray[np.float64]:
    """
    Возвращает матрицу R_axes (3×3), преобразующую координаты из LiDAR‐системы
    (x→вперёд, y→влево, z→вверх) в «сырые» координаты камеры KITTI
    (x→вправо, y→вниз, z→вперёд).

    Если хочется перевести точку P_lidar (LiDAR‐координаты) в P_cam_axes
    без учёта extrinsics (только изменение ориентации осей), то:
        P_cam_axes = R_axes @ P_lidar

    Возвращаемая матрица:
        [[ 0, -1,  0],
         [ 0,  0, -1],
         [ 1,  0,  0]]
    """
    return np.array([
        [0, -1,  0],
        [0,  0, -1],
        [1,  0,  0]
    ], dtype=np.float64)
