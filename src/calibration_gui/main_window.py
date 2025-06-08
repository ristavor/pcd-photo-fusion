"""
Main window of the calibration application.
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QApplication, QStackedWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
import os
from pathlib import Path
from .calibration_wizard import CalibrationWizard
from .results_window import ResultsWindow
from .processing_selection_window import ProcessingSelectionWindow
from .rectification_window import RectificationWindow
from .colorization_window import ColorizationWindow

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("KITTI Data Processing Tool")
        
        # Set application icon
        icon_path = Path(__file__).parent / "resources" / "icons" / "app_icon.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
        # Set window size
        self.resize(800, 600)
        
        # Create stacked widget for managing different screens
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Create and add start screen
        self.start_screen = QWidget()
        start_layout = QVBoxLayout(self.start_screen)
        
        # Create buttons
        self.calibration_btn = QPushButton("Start Calibration")
        self.calibration_btn.setMinimumHeight(50)
        self.calibration_btn.clicked.connect(self.start_calibration)
        
        self.processing_btn = QPushButton("Process Data")
        self.processing_btn.setMinimumHeight(50)
        self.processing_btn.clicked.connect(self.start_processing)
        
        # Add buttons to layout
        start_layout.addWidget(self.calibration_btn)
        start_layout.addWidget(self.processing_btn)
        start_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add start screen to stacked widget
        self.stacked_widget.addWidget(self.start_screen)
        
        # Initialize other screens (will be created when needed)
        self.wizard = None
        self.results_window = None
        self.processing_selection = None
        self.rectification_window = None
        self.colorization_window = None
    
    def start_calibration(self):
        """Start the calibration wizard."""
        if self.wizard is None:
            self.wizard = CalibrationWizard(self)
            self.stacked_widget.addWidget(self.wizard)
        self.stacked_widget.setCurrentWidget(self.wizard)
    
    def start_processing(self):
        """Start the processing selection screen."""
        if self.processing_selection is None:
            self.processing_selection = ProcessingSelectionWindow(self)
            self.stacked_widget.addWidget(self.processing_selection)
        self.stacked_widget.setCurrentWidget(self.processing_selection)
    
    def show_rectification(self):
        """Show the rectification window."""
        if self.rectification_window is None:
            self.rectification_window = RectificationWindow(self)
            self.stacked_widget.addWidget(self.rectification_window)
        self.stacked_widget.setCurrentWidget(self.rectification_window)
    
    def show_colorization(self):
        """Show the colorization window."""
        if self.colorization_window is None:
            self.colorization_window = ColorizationWindow(self)
            self.stacked_widget.addWidget(self.colorization_window)
        self.stacked_widget.setCurrentWidget(self.colorization_window)
    
    def show_results(self, R_matrix, T_vector):
        """Show results window."""
        print(f"Debug: Showing results with R_matrix shape: {R_matrix.shape}, T_vector shape: {T_vector.shape}")
        
        # Create new results window each time to ensure fresh state
        self.results_window = ResultsWindow(R_matrix, T_vector, self)
        self.stacked_widget.addWidget(self.results_window)
        self.stacked_widget.setCurrentWidget(self.results_window)
        
        # Force update
        self.results_window.show()
        self.results_window.raise_()
    
    def show_start_screen(self):
        """Show the start screen."""
        self.stacked_widget.setCurrentWidget(self.start_screen)
    
    def show_previous(self):
        """Show the processing selection screen."""
        if self.processing_selection:
            self.stacked_widget.setCurrentWidget(self.processing_selection)
    
    def closeEvent(self, event):
        """Handle window close event."""
        QApplication.instance().quit()
        event.accept()

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec() 