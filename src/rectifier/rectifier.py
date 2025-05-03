# src/rectifier/rectifier.py
import cv2
import numpy as np
from .calib import read_kitti_cam_calib

class ImageRectifier:
    def __init__(self,
                 calib_path: str,
                 cam_idx:    int,
                 image_size: tuple[int,int]):
        """
        calib_path: путь к calib_cam_to_cam.txt
        cam_idx:    номер камеры (0..3)
        image_size: (width, height) исходного кадра
        """
        # 1) Читаем все параметры
        K, D, R_rect, P_rect = read_kitti_cam_calib(calib_path, cam_idx)
        # 2) Предвычисляем карты remap
        w, h = image_size
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=K,
            distCoeffs=   D,
            R=            R_rect,
            newCameraMatrix=P_rect[:, :3],
            size=(w, h),
            m1type=cv2.CV_16SC2
        )

    def rectify(self, img: np.ndarray) -> np.ndarray:
        """
        Возвращает undistort+rectified изображение того же размера.
        """
        return cv2.remap(
            src=img,
            map1=self.map1,
            map2=self.map2,
            interpolation=cv2.INTER_LINEAR
        )
