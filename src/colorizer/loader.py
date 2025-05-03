import cv2
import numpy as np

def load_image(path: str) -> np.ndarray:
    """
    Читает PNG/JPEG-картинку и возвращает NumPy-матрицу (H×W×3).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    return img

def load_velodyne(path: str) -> np.ndarray:
    """
    Читает .bin Velodyne-файл и возвращает массив (N,4): X,Y,Z,intensity.
    """
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Неправильный формат Velodyne-файла: {path}")
    return pts.reshape(-1, 4)
