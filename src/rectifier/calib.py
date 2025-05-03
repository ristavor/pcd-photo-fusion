import numpy as np

def read_cam_to_cam(path: str) -> dict:
    """
    Считывает весь файл calib_cam_to_cam.txt в словарь:
      ключи — строки до «:»,
      значения — NumPy-массивы.
    """
    data = {}
    with open(path, 'r') as f:
        for line in f:
            key, vals = line.split(':', 1)
            data[key.strip()] = np.fromstring(vals, sep=' ')
    return data

def read_kitti_cam_calib(path: str, cam_idx: int):
    """
    Берёт из результата read_cam_to_cam нужные матрицы для камеры #cam_idx:
      K      — 3×3,
      D      — (5,),
      R_rect — 3×3,
      P_rect — 3×4.
    """
    d = read_cam_to_cam(path)
    idx = f'{cam_idx:02d}'

    K      = d[f'K_{idx}'].reshape(3,3)
    D      = d[f'D_{idx}']               # 5 коэффициентов
    R_rect = d[f'R_rect_{idx}'].reshape(3,3)
    P_rect = d[f'P_rect_{idx}'].reshape(3,4)

    return K, D, R_rect, P_rect
