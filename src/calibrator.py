import numpy as np

def read_cam_to_cam(path):
    data = {}
    with open(path) as f:
        for line in f:
            key, vals = line.split(':',1)
            data[key] = np.fromstring(vals, sep=' ')
    # P0 = data['P0'].reshape(3,4)
    return data

def read_velo_to_cam(path):
    data = {}
    with open(path) as f:
        for line in f:
            if ':' not in line:
                continue
            key, vals = line.split(':', 1)
            data[key.strip()] = np.fromstring(vals, sep=' ')
    # Собираем R (3×3) и T (3,)
    R = data['R'].reshape(3, 3)
    T = data['T']
    return R, T


