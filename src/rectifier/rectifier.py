from pathlib import Path
import cv2
import numpy as np

from .calib import (
    read_kitti_cam_calib,
    get_full_image_size,
    get_rectified_size,
)

class ImageRectifier:
    """
    Делает undistort + rectify + crop KITTI-изображение:
      1) читает параметры K, D, R_rect, P_rect
      2) initUndistortRectifyMap → remap
      3) обрезает по S_rect_0X (верхний левый угол)
    """

    def __init__(
        self,
        calib_cam_path: Path | str,
        cam_idx: int,
        interp: int = cv2.INTER_LINEAR
    ) -> None:
        """
        :param calib_cam_path: путь к calib_cam_to_cam.txt
        :param cam_idx:        номер камеры (0..3)
        :param interp:         метод интерполяции для remap
        """
        calib_cam_path = str(calib_cam_path)

        # 1) Считываем матрицы из калибровки
        K, D, R_rect, P_rect = read_kitti_cam_calib(calib_cam_path, cam_idx)

        # 2) Получаем размеры:
        #    полный выпрямлённый кадр S_0X и итоговый кроп S_rect_0X
        w_full, h_full     = get_full_image_size(calib_cam_path, cam_idx)
        self.w_crop, self.h_crop = get_rectified_size(calib_cam_path, cam_idx)

        # 3) Внутренняя матрица для remap — это P_rect[:3,:3]
        self.K       = K
        self.D       = D
        self.R_rect  = R_rect
        self.P_new   = P_rect[:3, :3]

        # Параметры для lazy-инициализации remap-карт
        self._map    = None
        self._params = dict(size=(w_full, h_full), interp=interp)

    def _init_map(self):
        """
        Ленивый расчёт карт undistort+rectify (map1, map2).
        """
        if self._map is None:
            w_full, h_full = self._params['size']
            self._map = cv2.initUndistortRectifyMap(
                cameraMatrix    = self.K,
                distCoeffs      = self.D,
                R               = self.R_rect,
                newCameraMatrix = self.P_new,
                size            = (w_full, h_full),
                m1type          = cv2.CV_32FC1,
            )
        return self._map

    def rectify(self, img: np.ndarray) -> np.ndarray:
        """
        1) remap: undistort + rectify
        2) crop по S_rect_0X (верхний-левый угол)
        :param img: uint8 BGR H×W×3
        :return:    rectified & cropped uint8 BGR
        """
        map1, map2 = self._init_map()
        full = cv2.remap(
            img,
            map1,
            map2,
            interpolation=self._params['interp']
        )
        # жёсткий crop от (0,0) до (w_crop, h_crop)
        return full[0 : self.h_crop, 0 : self.w_crop]
