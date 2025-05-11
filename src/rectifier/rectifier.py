from functools import cached_property
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from .calib import (
    read_kitti_cam_calib,
    get_full_image_size,
    get_rectified_size,
)


class ImageRectifier:
    """
    Undistort + rectify + crop KITTI-изображение.
    """

    def __init__(
            self,
            calib_cam_path: Path | str,
            cam_idx: int,
            interp: int = cv2.INTER_LINEAR
    ) -> None:
        calib_cam_path = str(calib_cam_path)
        K, D, R_rect, P_rect = read_kitti_cam_calib(calib_cam_path, cam_idx)

        w_full, h_full = get_full_image_size(calib_cam_path, cam_idx)
        w_crop, h_crop = get_rectified_size(calib_cam_path, cam_idx)

        assert isinstance(K, np.ndarray) and K.shape == (3, 3)
        assert isinstance(D, np.ndarray)
        assert isinstance(R_rect, np.ndarray) and R_rect.shape == (3, 3)
        assert isinstance(P_rect, np.ndarray) and P_rect.shape == (3, 4)

        self.K = K
        self.D = D
        self.R_rect = R_rect
        self.P_new = P_rect[:3, :3]
        self._full_size = (w_full, h_full)
        self.w_crop, self.h_crop = w_crop, h_crop
        self._interp = interp

    @cached_property
    def _remap_map(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Инициализирует и кеширует карты remap.
        """
        map1, map2 = cv2.initUndistortRectifyMap(
            cameraMatrix=self.K,
            distCoeffs=self.D,
            R=self.R_rect,
            newCameraMatrix=self.P_new,
            size=self._full_size,
            m1type=cv2.CV_32FC1,
        )
        return map1, map2

    def rectify(self, img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        Undistort + rectify + crop.
        """
        assert isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] in (3, 4), \
            "img must be H×W×3 or H×W×4 BGR image"
        map1, map2 = self._remap_map
        full = cv2.remap(img, map1, map2, interpolation=self._interp)
        return full[0:self.h_crop, 0:self.w_crop]
