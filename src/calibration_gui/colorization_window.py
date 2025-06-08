from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QSpinBox, QMessageBox,
    QCheckBox
)
from PyQt6.QtCore import Qt
from pathlib import Path
from colorizer import Colorizer, read_velo_to_cam
from rectifier import ImageRectifier, read_kitti_cam_calib
from calibration.calib_io import create_ideal_camera_params
import cv2
import numpy as np
import open3d as o3d

# --- Utility to read P_new (rectified K) from cam_to_cam.txt ---
def read_rectified_K(cam_calib_path, cam_idx):
    """Read P_rect_0X from KITTI cam_to_cam.txt for the given camera index."""
    with open(cam_calib_path, 'r') as f:
        for line in f:
            if line.startswith(f'P_rect_{cam_idx:02d}'):
                vals = line.strip().split()[1:]
                P = np.array([float(x) for x in vals], dtype=np.float64).reshape(3, 4)
                return P[:3, :3]
    raise ValueError(f"P_rect_{cam_idx:02d} not found in {cam_calib_path}")

class ColorizationWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.colored_pcd = None
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Image file selection
        img_layout = QHBoxLayout()
        self.img_label = QLabel("Image file:")
        self.img_path_label = QLabel("No file selected")
        self.img_btn = QPushButton("Browse")
        self.img_btn.clicked.connect(self.select_image)
        img_layout.addWidget(self.img_label)
        img_layout.addWidget(self.img_path_label)
        img_layout.addWidget(self.img_btn)
        layout.addLayout(img_layout)

        # Image rectification checkbox
        self.rectified_checkbox = QCheckBox("Image is already rectified")
        layout.addWidget(self.rectified_checkbox)
        
        # Point cloud file selection
        pcd_layout = QHBoxLayout()
        self.pcd_label = QLabel("Point cloud file:")
        self.pcd_path_label = QLabel("No file selected")
        self.pcd_btn = QPushButton("Browse")
        self.pcd_btn.clicked.connect(self.select_point_cloud)
        pcd_layout.addWidget(self.pcd_label)
        pcd_layout.addWidget(self.pcd_path_label)
        pcd_layout.addWidget(self.pcd_btn)
        layout.addLayout(pcd_layout)
        
        # KITTI dataset checkbox
        self.kitti_checkbox = QCheckBox("KITTI dataset")
        self.kitti_checkbox.stateChanged.connect(self.on_kitti_changed)
        layout.addWidget(self.kitti_checkbox)
        
        # Camera calibration file selection (now optional)
        cam_calib_layout = QHBoxLayout()
        self.cam_calib_label = QLabel("Camera calibration file (optional):")
        self.cam_calib_path_label = QLabel("No file selected")
        self.cam_calib_btn = QPushButton("Browse")
        self.cam_calib_btn.clicked.connect(self.select_camera_calibration)
        cam_calib_layout.addWidget(self.cam_calib_label)
        cam_calib_layout.addWidget(self.cam_calib_path_label)
        cam_calib_layout.addWidget(self.cam_calib_btn)
        layout.addLayout(cam_calib_layout)
        
        # Velodyne calibration file selection
        velo_calib_layout = QHBoxLayout()
        self.velo_calib_label = QLabel("Velodyne calibration file:")
        self.velo_calib_path_label = QLabel("No file selected")
        self.velo_calib_btn = QPushButton("Browse")
        self.velo_calib_btn.clicked.connect(self.select_velodyne_calibration)
        velo_calib_layout.addWidget(self.velo_calib_label)
        velo_calib_layout.addWidget(self.velo_calib_path_label)
        velo_calib_layout.addWidget(self.velo_calib_btn)
        layout.addLayout(velo_calib_layout)
        
        # Camera index selection
        cam_layout = QHBoxLayout()
        self.cam_label = QLabel("Camera index:")
        self.cam_spinbox = QSpinBox()
        self.cam_spinbox.setRange(0, 3)  # KITTI has cameras 0-3
        self.cam_spinbox.setEnabled(False)  # Initially disabled
        cam_layout.addWidget(self.cam_label)
        cam_layout.addWidget(self.cam_spinbox)
        layout.addLayout(cam_layout)
        
        # Process button
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_colorization)
        layout.addWidget(self.process_btn)
        
        # View Result button (initially hidden)
        self.view_result_btn = QPushButton("View Result")
        self.view_result_btn.clicked.connect(self.view_result)
        self.view_result_btn.setEnabled(False)
        layout.addWidget(self.view_result_btn)
        
        # Save as TXT button (initially hidden)
        self.save_txt_btn = QPushButton("Save as KITTI TXT")
        self.save_txt_btn.clicked.connect(self.save_result_txt)
        self.save_txt_btn.setEnabled(False)
        layout.addWidget(self.save_txt_btn)
        
        # Back button
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.go_back)
        layout.addWidget(self.back_btn)
        
    def on_kitti_changed(self, state):
        """Handle KITTI checkbox state change."""
        is_kitti = state == Qt.CheckState.Checked.value
        self.cam_spinbox.setEnabled(is_kitti)
        
    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.img_path_label.setText(file_name)
            
    def select_point_cloud(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Point Cloud", "", "Point Cloud Files (*.bin *.txt *.pcd)"
        )
        if file_name:
            self.pcd_path_label.setText(file_name)
            
    def select_camera_calibration(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Camera Calibration", "", "Text Files (*.txt)"
        )
        if file_name:
            self.cam_calib_path_label.setText(file_name)
            
    def select_velodyne_calibration(self):
        is_kitti = self.kitti_checkbox.isChecked()
        if is_kitti:
            file_filter = "Text Files (*.txt)"
        else:
            file_filter = "JSON Files (*.json)"
            
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Velodyne Calibration", "", file_filter
        )
        if file_name:
            self.velo_calib_path_label.setText(file_name)
            
    def process_colorization(self):
        # Check if required files are selected
        if any(label.text() == "No file selected" for label in [
            self.img_path_label, self.pcd_path_label,
            self.velo_calib_path_label
        ]):
            QMessageBox.warning(self, "Error", "Please select all required files")
            return
            
        try:
            print("\n[Colorizer GUI] --- Начало процесса ---")
            print("Image file:", self.img_path_label.text())
            print("Point cloud file:", self.pcd_path_label.text())
            print("Camera calibration file:", self.cam_calib_path_label.text())
            print("Velodyne calibration file:", self.velo_calib_path_label.text())
            print("Camera index:", self.cam_spinbox.value())
            print("Is KITTI dataset:", self.kitti_checkbox.isChecked())
            
            # Load image
            img = cv2.imread(self.img_path_label.text())
            if img is None:
                print("[ERROR] Не удалось загрузить изображение!")
                raise RuntimeError("Image not loaded")
            print("Image shape:", img.shape)

            # Get camera matrix K
            cam_idx = self.cam_spinbox.value()
            cam_calib_path = self.cam_calib_path_label.text()
            is_kitti = self.kitti_checkbox.isChecked()
            
            if cam_calib_path == "No file selected":
                # Create ideal camera parameters if no calibration file provided
                print("No camera calibration file provided, using ideal parameters")
                K, _ = create_ideal_camera_params((img.shape[1], img.shape[0]))
                if not self.rectified_checkbox.isChecked():
                    # For ideal parameters, we don't need rectification
                    print("Using ideal camera parameters (no rectification needed)")
                else:
                    print("Using ideal camera parameters (image marked as rectified)")
            else:
                if not is_kitti:
                    raise ValueError("Camera calibration file is only supported for KITTI dataset")
                    
                if not self.rectified_checkbox.isChecked():
                    # Rectify image if not already rectified
                    rectifier = ImageRectifier(
                        calib_cam_path=cam_calib_path,
                        cam_idx=cam_idx
                    )
                    img = rectifier.rectify(img)
                    K = rectifier.P_new
                    print("Image rectified, new shape:", img.shape)
                else:
                    # Use P_rect from calibration file directly
                    K = read_rectified_K(cam_calib_path, cam_idx)
                    print("Using existing rectified image")
            
            print("Camera matrix K:\n", K)
            
            # Load point cloud using the loader module
            from colorizer.loader import load_point_cloud
            pts = load_point_cloud(self.pcd_path_label.text())
            if pts.shape[1] > 3:
                pts = pts[:, :3]
            print("Point cloud shape:", pts.shape)
            
            # Get calibration matrices
            R, T = read_velo_to_cam(self.velo_calib_path_label.text(), is_kitti=is_kitti)
            print("R (velo_to_cam):\n", R)
            print("T (velo_to_cam):", T)
            
            # Colorize point cloud
            colorizer = Colorizer(R, T, K)
            self.colored_pcd = colorizer.colorize(pts, img)
            print("[Colorizer GUI] --- Окрашивание завершено ---\n")
            
            # Show completion message and enable view/save buttons
            QMessageBox.information(self, "Success", "Colorization completed successfully!")
            self.view_result_btn.setEnabled(True)
            self.save_txt_btn.setEnabled(True)
            
        except Exception as e:
            print("[Colorizer GUI] --- Ошибка ---", str(e))
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
            
    def view_result(self):
        if self.colored_pcd is None:
            QMessageBox.warning(self, "Error", "No colored point cloud to view!")
            return
        o3d.visualization.draw_geometries(
            [self.colored_pcd],
            window_name='Colored Point Cloud',
            width=800,
            height=600,
            point_show_normal=False
        )
            
    def save_result_txt(self):
        if self.colored_pcd is None:
            QMessageBox.warning(self, "Error", "No colored point cloud to save!")
            return
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Colored Point Cloud as TXT", "", "TXT Files (*.txt)"
        )
        if file_name:
            self._save_point_cloud_txt(file_name, self.colored_pcd)
            QMessageBox.information(self, "Success", "Point cloud saved as TXT!")

    @staticmethod
    def _save_point_cloud_txt(filename, pcd):
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        colors_uint8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
        data = np.hstack([points, colors_uint8])
        np.savetxt(filename, data, fmt="%.6f %.6f %.6f %d %d %d")

    def go_back(self):
        if self.parent():
            # Get the main window instance
            main_window = self.parent().parent()
            if main_window and hasattr(main_window, 'show_previous'):
                main_window.show_previous() 