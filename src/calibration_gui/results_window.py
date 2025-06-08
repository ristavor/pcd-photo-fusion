"""
Results window for displaying and saving calibration results.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QFileDialog,
    QMessageBox, QMainWindow
)
from PyQt6.QtCore import Qt
import json
import numpy as np

class ResultsWindow(QWidget):
    def __init__(self, R_matrix, T_vector, parent=None):
        super().__init__(parent)
        print(f"Debug: ResultsWindow initialized with R shape: {R_matrix.shape}, T shape: {T_vector.shape}")
        
        # Store reference to main window
        self.main_window = self.window()
        
        # Store results
        self.R_matrix = R_matrix
        self.T_vector = T_vector
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Add results display
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.update_results_display()
        layout.addWidget(self.results_text)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        self.save_btn = QPushButton("Save Results")
        self.save_btn.clicked.connect(self.save_results)
        button_layout.addWidget(self.save_btn)
        
        self.new_calib_btn = QPushButton("New Calibration")
        self.new_calib_btn.clicked.connect(self.start_new_calibration)
        button_layout.addWidget(self.new_calib_btn)
        
        layout.addLayout(button_layout)
        
        # Force immediate update
        self.update()
    
    def update_results(self, R_matrix, T_vector):
        """Update the results with new values."""
        print(f"Debug: Updating results with R shape: {R_matrix.shape}, T shape: {T_vector.shape}")
        self.R_matrix = R_matrix
        self.T_vector = T_vector
        self.update_results_display()
        self.update()
    
    def update_results_display(self):
        """Update the text display with current results."""
        print("Debug: Updating results display")
        text = "Calibration Results:\n\n"
        text += "Rotation Matrix (R):\n"
        text += str(self.R_matrix) + "\n\n"
        text += "Translation Vector (T):\n"
        text += str(self.T_vector)
        self.results_text.setText(text)
        self.results_text.update()
    
    def showEvent(self, event):
        """Handle show event."""
        super().showEvent(event)
        print("Debug: ResultsWindow shown")
        self.update_results_display()
        self.update()
    
    def save_results(self):
        """Save results to a JSON file."""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Calibration Results",
            "",
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                # Convert numpy arrays to lists for JSON serialization
                results = {
                    "R_matrix": self.R_matrix.tolist(),
                    "T_vector": self.T_vector.tolist()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(results, f, indent=4)
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Results saved to {file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to save results: {str(e)}"
                )
    
    def start_new_calibration(self):
        """Start a new calibration process."""
        if isinstance(self.main_window, QMainWindow):
            self.main_window.show_start_screen() 