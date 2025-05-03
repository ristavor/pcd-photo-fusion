import numpy as np

def read_cam_to_cam(path: str) -> dict:
    """
    Считывает параметры камеры из calib_cam_to_cam.txt.
    Возвращает словарь, ключи — строки до «:», а значения — np.array.
    """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            key, vals = line.split(':', 1)
            data[key.strip()] = np.fromstring(vals, sep=' ')
    return data

def read_velo_to_cam(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Считывает параметры R и T из calib_velo_to_cam.txt.
    Возвращает R (3×3) и T (3,).
    """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            if ':' not in line:
                continue
            key, vals = line.split(':', 1)
            data[key.strip()] = np.fromstring(vals, sep=' ')
    R = data['R'].reshape(3, 3)
    T = data['T']
    return R, T
