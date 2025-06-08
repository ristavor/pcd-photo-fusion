from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import Qt

class ProcessingSelectionWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Create buttons
        self.rectify_btn = QPushButton("Rectify Image")
        self.rectify_btn.setMinimumHeight(50)
        self.rectify_btn.clicked.connect(self.start_rectification)
        
        self.colorize_btn = QPushButton("Colorize Point Cloud")
        self.colorize_btn.setMinimumHeight(50)
        self.colorize_btn.clicked.connect(self.start_colorization)
        
        self.back_btn = QPushButton("Back")
        self.back_btn.setMinimumHeight(50)
        self.back_btn.clicked.connect(self.go_back)
        
        # Add buttons to layout
        layout.addWidget(self.rectify_btn)
        layout.addWidget(self.colorize_btn)
        layout.addWidget(self.back_btn)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
    def start_rectification(self):
        # Get the main window by traversing up the widget hierarchy
        main_window = self.window()
        if main_window:
            main_window.show_rectification()
            
    def start_colorization(self):
        # Get the main window by traversing up the widget hierarchy
        main_window = self.window()
        if main_window:
            main_window.show_colorization()
            
    def go_back(self):
        # Get the main window by traversing up the widget hierarchy
        main_window = self.window()
        if main_window:
            main_window.show_start_screen() 