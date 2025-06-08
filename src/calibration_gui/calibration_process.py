"""
Calibration process handler for the GUI.
"""

import cv2
import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict, Any
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from calibration.image_corners import (
    detect_image_corners,
    adjust_corners_interactively
)
from calibration.pcd_roi import (
    load_point_cloud,
    select_pointcloud_roi,
    extract_roi_cloud
)
from calibration.board_geometry import (
    compute_board_frame,
    generate_object_points
)
from calibration.calib_io import (
    load_camera_params
)
from calibration.viz_utils import (
    draw_overlay,
    make_overlay_image
)
from colorizer import Colorizer

class CalibrationProcess:
    def __init__(self):
        self.K = None  # Camera matrix
        self.D = None  # Distortion coefficients
        self.R_gt = None  # Ground truth rotation
        self.T_gt = None  # Ground truth translation
        
        # Current calibration state
        self.corners2d = None
        self.origin = None
        self.x_axis = None
        self.y_axis = None
        self.R_best = None
        self.T_best = None
        self.pcd = None  # Store point cloud object
        self.img = None  # Store image
        
        # Visualization state
        self.vis = None
        self.colored_window_open = False
    
    def load_camera_params(self, calib_path: str, cam_idx: int = 0) -> bool:
        """Load camera parameters from calibration file."""
        try:
            logger.info(f"Loading camera parameters from {calib_path}, camera index {cam_idx}")
            self.K, self.D, self.R_gt, self.T_gt = load_camera_params(calib_path, cam_idx)
            logger.debug(f"Loaded camera matrix K:\n{self.K}")
            logger.debug(f"Loaded distortion coefficients D:\n{self.D}")
            return True
        except Exception as e:
            logger.error(f"Error loading camera parameters: {e}")
            return False
    
    def process_image(self, image_path: str, pattern: Tuple[int, int]) -> bool:
        """Process image to detect chessboard corners."""
        try:
            logger.info(f"Processing image: {image_path}")
            self.img = cv2.imread(image_path)
            if self.img is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Let user select ROI
            x, y, w, h = map(int, cv2.selectROI("ROI", self.img))
            cv2.destroyWindow("ROI")
            if w == 0 or h == 0:
                logger.warning("No ROI selected")
                return False
            
            logger.debug(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")
            roi_img = self.img[y:y + h, x:x + w]
            self.corners2d = detect_image_corners(roi_img, pattern)
            self.corners2d += np.array([x, y], dtype=np.float32)
            self.corners2d = adjust_corners_interactively(self.img, self.corners2d, pattern)
            logger.info(f"Found {len(self.corners2d)} corners")
            return True
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return False
    
    def process_pointcloud(self, pcd_path: str) -> bool:
        """Process point cloud to detect chessboard."""
        try:
            logger.info(f"Processing point cloud: {pcd_path}")
            self.pcd = load_point_cloud(pcd_path)
            if self.pcd is None:
                raise ValueError(f"Could not load point cloud: {pcd_path}")
            
            logger.debug(f"Point cloud loaded, number of points: {len(self.pcd.points)}")
            idxs = select_pointcloud_roi(self.pcd)
            if not idxs:
                logger.warning("No points selected in ROI")
                return False
            
            logger.debug(f"Selected {len(idxs)} points in ROI")
            board_roi, _ = extract_roi_cloud(self.pcd, idxs)
            if board_roi is None or len(board_roi.points) == 0:
                raise ValueError("No points selected in ROI")
            
            try:
                logger.debug("Computing board frame...")
                self.origin, self.x_axis, self.y_axis, _ = compute_board_frame(board_roi)
                logger.debug(f"Board frame computed:\nOrigin: {self.origin}\nX axis: {self.x_axis}\nY axis: {self.y_axis}")
            except ValueError as e:
                logger.error(f"Error computing board frame: {e}")
                print(f"Error computing board frame: {e}")
                print("Please try selecting a different ROI with more points that clearly form a plane")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error processing point cloud: {e}")
            return False
    
    def find_best_configuration(self, pattern: Tuple[int, int], square_size: float) -> bool:
        """Find the best configuration for the calibration."""
        logger.info("Starting best configuration search...")
        logger.debug(f"Pattern: {pattern}, Square size: {square_size}, type: {type(square_size)}")
        
        if not all([self.corners2d is not None, self.origin is not None,
                   self.x_axis is not None, self.y_axis is not None]):
            logger.error("Missing required data for configuration search")
            return False
        
        if square_size is None or square_size <= 0:
            logger.error(f"Invalid square size: {square_size}, type: {type(square_size)}")
            return False
        
        candidates = self._debug_pnp_axes(
            self.corners2d, self.origin, self.x_axis, self.y_axis,
            pattern, square_size
        )
        
        if not candidates:
            logger.error("No valid configurations found")
            return False
        
        # Let user choose the best configuration
        idx = 0
        n = len(candidates)
        cv2.namedWindow("ChooseConfig", cv2.WINDOW_NORMAL)
        
        while True:
            try:
                name, mre_px, R_est, T_est = candidates[idx]
                rvec, _ = cv2.Rodrigues(R_est)
                tvec = T_est.reshape(3, 1)
                
                overlay_img = make_overlay_image(
                    np.asarray(self.pcd.points), rvec, tvec, self.K, self.D, self.img
                )
                # Отображаем название конфигурации и MRE
                text = f"{name} (MRE: {mre_px:.2f} px)"
                cv2.putText(overlay_img, text, (10, 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.imshow("ChooseConfig", overlay_img)
                key = cv2.waitKey(0) & 0xFF
                
                if key in (13, 10):  # Enter
                    cv2.destroyWindow("ChooseConfig")
                    self.R_best = R_est
                    self.T_best = T_est
                    logger.info(f"Selected configuration: {name}")
                    return True
                elif key == ord('a'):
                    idx = (idx - 1) % n
                    logger.debug(f"Previous configuration: {candidates[idx][0]}")
                elif key == ord('d'):
                    idx = (idx + 1) % n
                    logger.debug(f"Next configuration: {candidates[idx][0]}")
                elif key == 27:  # ESC
                    cv2.destroyWindow("ChooseConfig")
                    logger.info("Configuration selection cancelled")
                    return False
            except Exception as e:
                logger.error(f"Error in configuration selection loop: {e}")
                return False
    
    def _show_colored_pointcloud(self, colorizer: Colorizer,
                               points: np.ndarray, image: np.ndarray,
                               rvec: np.ndarray, tvec: np.ndarray) -> None:
        """Show colored point cloud visualization."""
        # Update matrices in colorizer
        R_mat = cv2.Rodrigues(rvec)[0]
        colorizer.R = R_mat
        colorizer.T = tvec.flatten()
        
        # Create colored point cloud
        pcd = colorizer.colorize(points, image)
        
        # If visualizer exists, close it
        if self.vis is not None:
            self.vis.destroy_window()
        
        # Create new visualizer
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window("Colored Point Cloud", 800, 600)
        self.vis.add_geometry(pcd)
        
        # Setup camera
        bbox = pcd.get_axis_aligned_bounding_box()
        self.vis.get_view_control().set_zoom(0.8)
        self.vis.get_view_control().set_front([0, 0, -1])
        self.vis.get_view_control().set_lookat(bbox.get_center())
        self.vis.get_view_control().set_up([0, -1, 0])
        
        self.colored_window_open = True
        
        # Run visualization loop
        while self.colored_window_open:
            self.vis.poll_events()
            self.vis.update_renderer()
            
            if not self.vis.poll_events():
                self.colored_window_open = False
                self.vis.destroy_window()
                break
            
            key_cloud = self.vis.poll_events()
            if key_cloud == 27:  # ESC
                self.colored_window_open = False
                self.vis.destroy_window()
                break
            
            cv2.waitKey(10)

    def interactive_refine(self, all_lidar_points: np.ndarray, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Interactive refinement of the calibration."""
        if self.R_best is None or self.T_best is None:
            raise ValueError("No calibration configuration selected")
            
        rvec, _ = cv2.Rodrigues(self.R_best)
        tvec = self.T_best.reshape(3, 1)
        
        delta_ang = 0.01   # rotation step (≈0.57°)
        delta_t = 0.01     # translation step (≈1 cm)
        mouse_sens = 0.005  # mouse sensitivity
        
        dragging = False
        last_x = last_y = 0
        
        # Create colorizer for point cloud coloring
        colorizer = Colorizer(self.R_best, tvec.flatten(), self.K)
        
        def on_mouse(event, x, y, flags, _):
            nonlocal dragging, last_x, last_y, rvec
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                last_x, last_y = x, y
            elif event == cv2.EVENT_MOUSEMOVE and dragging:
                dx = x - last_x
                dy = y - last_y
                rvec[1] += dx * mouse_sens  # yaw
                rvec[0] += dy * mouse_sens  # pitch
                last_x, last_y = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False
        
        cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Overlay", on_mouse)
        
        while True:
            draw_overlay(all_lidar_points, rvec, tvec, self.K, self.D, image, window_name="Overlay")
            
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC
                break
            
            # Rotation R
            if   key == ord('w'): rvec[0] -= delta_ang
            elif key == ord('s'): rvec[0] += delta_ang
            elif key == ord('a'): rvec[1] -= delta_ang
            elif key == ord('d'): rvec[1] += delta_ang
            elif key == ord('q'): rvec[2] -= delta_ang
            elif key == ord('e'): rvec[2] += delta_ang
            
            # Translation T
            elif key == ord('i'): tvec[0] -= delta_t
            elif key == ord('k'): tvec[0] += delta_t
            elif key == ord('j'): tvec[1] -= delta_t
            elif key == ord('l'): tvec[1] += delta_t
            elif key == ord('u'): tvec[2] -= delta_t
            elif key == ord('o'): tvec[2] += delta_t
            
            # Show colored point cloud
            elif key == ord('m'):
                self._show_colored_pointcloud(colorizer, all_lidar_points, image, rvec, tvec)
        
        cv2.destroyWindow("Overlay")
        return rvec, tvec
    
    def _debug_pnp_axes(self, corners2d: np.ndarray, origin: np.ndarray,
                       x_axis: np.ndarray, y_axis: np.ndarray,
                       pattern: Tuple[int, int], square_size: float) -> list:
        """
        Возвращает список из возможных конфигураций PnP:
          [(name, mre_px, R_est, T_est), …]
        Пользователь затем выбирает нужный.
        """
        logger.info("Starting PnP axes debug...")
        logger.debug(f"Input parameters:\nPattern: {pattern}\nSquare size: {square_size}")
        logger.debug(f"Origin: {origin}\nX axis: {x_axis}\nY axis: {y_axis}")
        
        cols, rows = pattern
        specs: list[tuple[bool, int, int, str]] = []
        for swap in (False, True):
            for sx in (1, -1):
                for sy in (1, -1):
                    name = f"{'swap,' if swap else ''}{'+' if sx>0 else '-'}x,{'+' if sy>0 else '-'}y"
                    specs.append((swap, sx, sy, name))

        results: list[tuple[str, float, np.ndarray, np.ndarray]] = []
        for swap, sx, sy, name in specs:
            try:
                logger.debug(f"Trying configuration: {name}")
                if swap:
                    xa = y_axis * sx
                    ya = x_axis * sy
                else:
                    xa = x_axis * sx
                    ya = y_axis * sy

                pts3d = generate_object_points(origin, xa, ya, pattern, square_size)
                logger.debug(f"Generated {len(pts3d)} 3D points")
                
                ok, rvec, tvec = cv2.solvePnP(
                    objectPoints=pts3d.reshape(-1, 1, 3),
                    imagePoints=corners2d.reshape(-1, 1, 2),
                    cameraMatrix=self.K,
                    distCoeffs=self.D,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    logger.warning(f"PnP failed for configuration {name}")
                    continue

                R_est = cv2.Rodrigues(rvec)[0]
                T_est = tvec.flatten()
                
                # Вычисляем MRE (Mean Reprojection Error) в пикселях
                proj_pts, _ = cv2.projectPoints(
                    pts3d.reshape(-1, 1, 3),
                    rvec, tvec,
                    self.K, self.D
                )
                proj_pts = proj_pts.reshape(-1, 2)
                mre_px = np.mean(np.linalg.norm(proj_pts - corners2d, axis=1))
                
                logger.debug(f"Configuration {name} - MRE: {mre_px:.2f} px")
                results.append((name, mre_px, R_est.astype(np.float32), T_est.astype(np.float32)))
            except Exception as e:
                logger.error(f"Error processing configuration {name}: {e}")
                continue

        logger.info(f"Found {len(results)} valid configurations")
        return results
    
    @staticmethod
    def _rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
        """Compute rotation error in degrees."""
        R = R_est @ R_gt.T
        cos_val = (np.trace(R) - 1) / 2
        cos_val = np.clip(cos_val, -1.0, 1.0)
        return np.degrees(np.arccos(cos_val)) 