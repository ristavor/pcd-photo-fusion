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
from .calibration_wizard import CalibrationWizard
from .results_window import ResultsWindow
from pathlib import Path

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Tool")
        
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
        
        # Create calibration button
        self.calibration_btn = QPushButton("Start Calibration")
        self.calibration_btn.setMinimumHeight(50)
        self.calibration_btn.clicked.connect(self.start_calibration)
        
        # Add button to layout
        start_layout.addWidget(self.calibration_btn)
        start_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Add start screen to stacked widget
        self.stacked_widget.addWidget(self.start_screen)
        
        # Initialize other screens (will be created when needed)
        self.wizard = None
        self.results_window = None
    
    def start_calibration(self):
        """Start the calibration wizard."""
        if self.wizard is None:
            self.wizard = CalibrationWizard(self)
            self.stacked_widget.addWidget(self.wizard)
        self.stacked_widget.setCurrentWidget(self.wizard)
    
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
    
    def closeEvent(self, event):
        """Handle window close event."""
        QApplication.instance().quit()
        event.accept()

def main():
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec() 