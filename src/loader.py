# src/loader.py
import cv2
import numpy as np

def load_image(path):
    """
    Читает PNG-картинку по полному пути и возвращает массив NumPy (H×W×3 или H×W).
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    return img

def load_velodyne(path):
    """
    Читает бинарный файл Velodyne (.bin) по полному пути.
    Возвращает numpy-массив формы (N,4): X, Y, Z, intensity.
    """
    pts = np.fromfile(str(path), dtype=np.float32)
    if pts.size % 4 != 0:
        raise ValueError(f"Некорректный формат Velodyne-файла: {path}")
    pts = pts.reshape(-1, 4)
    return pts
