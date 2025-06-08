from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFileDialog, QSpinBox, QMessageBox
)
from PyQt6.QtCore import Qt
from pathlib import Path
from rectifier import ImageRectifier
import cv2
import numpy as np

class RectificationWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.rectified_image = None
        
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
        
        # Calibration file selection
        calib_layout = QHBoxLayout()
        self.calib_label = QLabel("Camera calibration file:")
        self.calib_path_label = QLabel("No file selected")
        self.calib_btn = QPushButton("Browse")
        self.calib_btn.clicked.connect(self.select_calibration)
        calib_layout.addWidget(self.calib_label)
        calib_layout.addWidget(self.calib_path_label)
        calib_layout.addWidget(self.calib_btn)
        layout.addLayout(calib_layout)
        
        # Camera index selection
        cam_layout = QHBoxLayout()
        self.cam_label = QLabel("Camera index:")
        self.cam_spinbox = QSpinBox()
        self.cam_spinbox.setRange(0, 3)  # KITTI has cameras 0-3
        cam_layout.addWidget(self.cam_label)
        cam_layout.addWidget(self.cam_spinbox)
        layout.addLayout(cam_layout)
        
        # Process button
        self.process_btn = QPushButton("Process")
        self.process_btn.clicked.connect(self.process_image)
        layout.addWidget(self.process_btn)
        
        # Back button
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.go_back)
        layout.addWidget(self.back_btn)
        
    def select_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)"
        )
        if file_name:
            self.img_path_label.setText(file_name)
            
    def select_calibration(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration File", "", "Text Files (*.txt)"
        )
        if file_name:
            self.calib_path_label.setText(file_name)
            
    def process_image(self):
        if self.img_path_label.text() == "No file selected":
            QMessageBox.warning(self, "Error", "Please select an image file")
            return
        if self.calib_path_label.text() == "No file selected":
            QMessageBox.warning(self, "Error", "Please select a calibration file")
            return
            
        try:
            # Load and process image
            img = cv2.imread(self.img_path_label.text())
            rectifier = ImageRectifier(
                calib_cam_path=self.calib_path_label.text(),
                cam_idx=self.cam_spinbox.value()
            )
            self.rectified_image = rectifier.rectify(img)
            
            # Show success message
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Icon.Information)
            msg.setText("Processing complete!")
            msg.setInformativeText("Would you like to save the result?")
            msg.setStandardButtons(
                QMessageBox.StandardButton.Save | 
                QMessageBox.StandardButton.Cancel
            )
            
            if msg.exec() == QMessageBox.StandardButton.Save:
                self.save_result()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
            
    def save_result(self):
        if self.rectified_image is None:
            return
            
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Rectified Image", "", "PNG Files (*.png)"
        )
        if file_name:
            cv2.imwrite(file_name, self.rectified_image)
            QMessageBox.information(self, "Success", "Image saved successfully!")
            
    def go_back(self):
        if self.parent():
            # Get the main window instance
            main_window = self.parent().parent()
            if main_window and hasattr(main_window, 'show_previous'):
                main_window.show_previous() 