import cv2
import numpy as np

from .calib import read_kitti_cam_calib


class ImageRectifier:
    """
    Делает undistort+rectify + обрезку по S_rect_0X.
    """

    def __init__(self, calib_cam_path: str, cam_idx: int):
        # 1) Считаем K, D, R_rect, P_rect
        K, D, R_rect, P_rect = read_kitti_cam_calib(calib_cam_path, cam_idx)
        self.K       = K
        self.D       = D
        self.R_rect  = R_rect
        # P_rect[:3,:3] — та же K*R_rect, но remap ждёт «новую» внутр. матрицу
        self.P_new   = P_rect[:3, :3]

        # 2) Размер полного выпрямлённого кадра (S_0X)
        all_size = read_kitti_cam_calib.__globals__['read_cam_to_cam'](calib_cam_path)[f'S_0{cam_idx}']
        h_full, w_full = int(all_size[1]), int(all_size[0])

        # 3) Размер кропа (S_rect_0X)
        rect_size = read_kitti_cam_calib.__globals__['read_cam_to_cam'](calib_cam_path)[f'S_rect_0{cam_idx}']
        self.w_crop, self.h_crop = int(rect_size[0]), int(rect_size[1])

        # 4) Заранее считаем карты remap
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            cameraMatrix   = self.K,
            distCoeffs     = self.D,
            R              = self.R_rect,
            newCameraMatrix= self.P_new,
            size           = (w_full, h_full),
            m1type         = cv2.CV_32FC1,
        )

    def rectify(self, img: np.ndarray) -> np.ndarray:
        """
        1) remap: undistort + rectify
        2) crop по верхнему-левому углу до (w_crop, h_crop)
        """
        full = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR)
        return full[0:self.h_crop, 0:self.w_crop]
