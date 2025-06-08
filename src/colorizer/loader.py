import cv2
import numpy as np
from pathlib import Path


def load_image(path: str) -> np.ndarray:
    """
    Читает PNG/JPEG-картинку и возвращает NumPy-матрицу (H×W×3).
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    return img


def load_point_cloud(path: str) -> np.ndarray:
    """
    Загружает облако точек из файла.

    Поддерживаемые форматы:
      - .pcd: читаем как текстовый файл, пропускаем заголовок и строки с NaN.
      - .bin : Velodyne Binary → читаем float32, reshape(-1,4), берём первые три столбца.
      - .txt : текстовый файл XYZ или XYZI → np.loadtxt, берём первые три столбца.

    Параметры:
      path (str): путь к файлу облака точек.

    Возвращает:
      np.ndarray: массив точек (N,3) или (N,4) в зависимости от формата.

    Исключения:
      ValueError: если формат файла не поддерживается.
      RuntimeError: если облако пустое после загрузки.
    """
    p = Path(path)
    ext = p.suffix.lower()

    if ext == '.pcd':
        # Читаем PCD файл как текстовый
        with open(p, 'r') as f:
            lines = f.readlines()
        
        # Находим начало данных
        data_start = 0
        for i, line in enumerate(lines):
            if line.strip() == 'DATA ascii':
                data_start = i + 1
                break
        
        # Читаем только строки с данными, пропуская заголовок
        data_lines = lines[data_start:]
        
        # Фильтруем строки с NaN
        valid_lines = []
        for line in data_lines:
            values = line.strip().split()
            if len(values) >= 3:  # Проверяем, что есть хотя бы X Y Z
                try:
                    x, y, z = map(float, values[:3])
                    if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                        valid_lines.append(f"{x} {y} {z}")
                except ValueError:
                    continue
        
        if not valid_lines:
            raise RuntimeError(f"Нет валидных точек в PCD файле: {path}")
        
        # Создаем временный файл с валидными точками
        temp_file = p.parent / f"{p.stem}_valid.txt"
        try:
            with open(temp_file, 'w') as f:
                f.write('\n'.join(valid_lines))
            
            # Загружаем точки из временного файла
            data = np.loadtxt(str(temp_file), dtype=np.float32)
            if data.ndim == 1:
                data = data.reshape(1, -1)
        finally:
            # Удаляем временный файл
            if temp_file.exists():
                temp_file.unlink()
                
    elif ext == '.bin':
        data = np.fromfile(str(p), dtype=np.float32)
        if data.size % 4 != 0:
            raise ValueError(f"Неправильный формат Velodyne‐файла: {path}")
        data = data.reshape(-1, 4)
    elif ext == '.txt':
        data = np.loadtxt(str(p), dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"Текстовый файл {path} должен содержать минимум три столбца (X Y Z).")
    else:
        raise ValueError(f"Неподдерживаемый формат облака точек: {ext}")

    if len(data) == 0:
        raise RuntimeError(f"Загруженное облако точек пустое: {path}")
    return data


def load_velodyne(path: str) -> np.ndarray:
    """
    Читает .bin Velodyne-файл и возвращает массив (N,4): X,Y,Z,intensity.
    """
    return load_point_cloud(path)  # Используем общую функцию загрузки
