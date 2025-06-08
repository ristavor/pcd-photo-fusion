"""
Calibration wizard window for parameter input and calibration process.
"""

from PyQt6.QtWidgets import (
    QWidget, QWizardPage, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QDoubleSpinBox,
    QFileDialog, QMessageBox, QProgressBar, QMainWindow
)
from PyQt6.QtCore import Qt, pyqtSignal, QLocale
import json
import os
import cv2
import numpy as np

from .calibration_process import CalibrationProcess
from .results_window import ResultsWindow

class ParametersPage(QWizardPage):
    def __init__(self):
        super().__init__()
        self.setTitle("Calibration Parameters")
        
        layout = QVBoxLayout()
        
        # Chessboard pattern
        pattern_layout = QHBoxLayout()
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(2, 20)
        self.cols_spin.setValue(7)
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(2, 20)
        self.rows_spin.setValue(5)
        pattern_layout.addWidget(QLabel("Chessboard pattern (cols, rows):"))
        pattern_layout.addWidget(self.cols_spin)
        pattern_layout.addWidget(self.rows_spin)
        layout.addLayout(pattern_layout)
        
        # Square size
        size_layout = QHBoxLayout()
        self.square_size = QDoubleSpinBox()
        self.square_size.setRange(0.01, 1.0)
        self.square_size.setValue(0.10)
        self.square_size.setSingleStep(0.01)
        self.square_size.setLocale(QLocale(QLocale.Language.English, QLocale.Country.UnitedStates))
        size_layout.addWidget(QLabel("Square size (meters):"))
        size_layout.addWidget(self.square_size)
        layout.addLayout(size_layout)
        
        # File paths
        self.image_path = self.create_file_input("Image file:", layout)
        self.pcd_path = self.create_file_input("Point cloud file:", layout)
        self.calib_path = self.create_file_input("Camera calibration file (optional):", layout)
        
        self.setLayout(layout)
        
        # Register fields
        self.registerField("cols", self.cols_spin, "value", self.cols_spin.valueChanged)
        self.registerField("rows", self.rows_spin, "value", self.rows_spin.valueChanged)
        self.registerField("square_size*", self.square_size, "value", self.square_size.valueChanged)
        self.registerField("image_path*", self.image_path)
        self.registerField("pcd_path*", self.pcd_path)
        self.registerField("calib_path", self.calib_path)  # Removed * to make it optional

        # Connect value changed signals to trigger validation
        self.cols_spin.valueChanged.connect(self._on_field_changed)
        self.rows_spin.valueChanged.connect(self._on_field_changed)
        self.square_size.valueChanged.connect(self._on_field_changed)
        self.image_path.textChanged.connect(self._on_field_changed)
        self.pcd_path.textChanged.connect(self._on_field_changed)
        self.calib_path.textChanged.connect(self._on_field_changed)

        # Trigger initial validation
        self._on_field_changed()
    
    def _on_field_changed(self):
        """Handle field changes and update page completion status."""
        self.completeChanged.emit()
    
    def isComplete(self):
        """Check if all required fields are filled."""
        return all([
            self.cols_spin.value() > 0,
            self.rows_spin.value() > 0,
            self.square_size.value() > 0,
            bool(self.image_path.text().strip()),
            bool(self.pcd_path.text().strip())
            # calib_path is optional, so we don't check it
        ])
    
    def create_file_input(self, label_text, parent_layout):
        """Create a file input field with browse button."""
        layout = QHBoxLayout()
        line_edit = QLineEdit()
        browse_btn = QPushButton("Browse")
        
        layout.addWidget(QLabel(label_text))
        layout.addWidget(line_edit)
        layout.addWidget(browse_btn)
        
        browse_btn.clicked.connect(lambda: self.browse_file(line_edit))
        parent_layout.addLayout(layout)
        
        return line_edit
    
    def browse_file(self, line_edit):
        """Open file dialog and set the selected file path."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select File",
            "",
            "All Files (*.*)"
        )
        if file_path:
            line_edit.setText(file_path)
            self._on_field_changed()
    
    def validatePage(self):
        """Validate that all required fields are filled."""
        if not self.isComplete():
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please fill in all required fields."
            )
            return False
        return True

class CalibrationWizard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Camera-LiDAR Calibration Wizard")
        
        # Store reference to main window
        self.main_window = self.window()
        
        # Initialize calibration process
        self.calib_process = CalibrationProcess()
        
        # Create main layout
        layout = QVBoxLayout(self)
        
        # Add parameters page
        self.parameters_page = ParametersPage()
        layout.addWidget(self.parameters_page)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.back_btn = QPushButton("Back")
        self.back_btn.clicked.connect(self.go_back)
        button_layout.addWidget(self.back_btn)
        
        self.next_btn = QPushButton("Start Calibration")
        self.next_btn.clicked.connect(self.start_calibration)
        button_layout.addWidget(self.next_btn)
        
        layout.addLayout(button_layout)
    
    def __del__(self):
        """Cleanup resources when the object is destroyed."""
        if hasattr(self, 'calib_process'):
            self.calib_process.cleanup()
    
    def go_back(self):
        """Return to start screen."""
        if isinstance(self.main_window, QMainWindow):
            self.calib_process.cleanup()
            self.main_window.show_start_screen()
    
    def start_calibration(self):
        """Start the calibration process."""
        if not self.parameters_page.isComplete():
            QMessageBox.warning(
                self,
                "Validation Error",
                "Please fill in all required fields."
            )
            return
        
        try:
            # Get parameters directly from spinboxes
            cols = self.parameters_page.cols_spin.value()
            rows = self.parameters_page.rows_spin.value()
            square_size = self.parameters_page.square_size.value()
            image_path = self.parameters_page.image_path.text().strip()
            pcd_path = self.parameters_page.pcd_path.text().strip()
            calib_path = self.parameters_page.calib_path.text().strip()
            
            # Validate square_size
            if square_size <= 0:
                raise ValueError(f"Invalid square size: {square_size}")
            
            params = {
                "pattern": (cols, rows),
                "square_size": float(square_size),
                "image_path": image_path,
                "pcd_path": pcd_path,
                "calib_path": calib_path
            }
            
            # Process image first
            if not self.calib_process.process_image(params["image_path"], params["pattern"]):
                raise RuntimeError("Failed to process image")
            
            # Load camera parameters (now that we have the image loaded)
            if not self.calib_process.load_camera_params(params["calib_path"]):
                raise RuntimeError("Failed to load camera parameters")
            
            # Process point cloud
            if not self.calib_process.process_pointcloud(params["pcd_path"]):
                raise RuntimeError("Failed to process point cloud")
            
            # Find best configuration
            if not self.calib_process.find_best_configuration(
                params["pattern"], params["square_size"]
            ):
                # User cancelled configuration selection
                self.calib_process.cleanup()
                return
            
            # Interactive refinement
            rvec_refined, tvec_refined = self.calib_process.interactive_refine(
                np.asarray(self.calib_process.pcd.points),
                cv2.imread(params["image_path"])
            )
            
            # Convert to matrices
            R_refined = cv2.Rodrigues(rvec_refined)[0]
            T_refined = tvec_refined.flatten()
            
            print(f"Debug: Calibration completed. R shape: {R_refined.shape}, T shape: {T_refined.shape}")
            
            # Show results window
            main_window = self.window()
            if isinstance(main_window, QMainWindow):
                print("Debug: Calling show_results on main window")
                self.calib_process.cleanup()  # Cleanup before showing results
                main_window.show_results(R_refined, T_refined)
            else:
                print(f"Debug: Main window not found. Window type: {type(main_window)}")
            
        except Exception as e:
            print(f"Debug: Calibration failed with error: {str(e)}")
            self.calib_process.cleanup()  # Cleanup on error
            QMessageBox.critical(
                self,
                "Error",
                f"Calibration failed: {str(e)}"
            ) 