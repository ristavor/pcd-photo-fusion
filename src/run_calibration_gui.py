#!/usr/bin/env python3

"""
Main entry point for running the calibration GUI.
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt
from calibration_gui.main_window import MainWindow

def set_light_theme(app: QApplication):
    """Set light theme for the application."""
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(233, 233, 233))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(255, 255, 255))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, QColor(0, 0, 255))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(0, 120, 215))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    
    app.setPalette(palette)
    
    # Set stylesheet for better button and checkbox visibility
    app.setStyleSheet("""
        QPushButton {
            border: 1px solid #999999;
            border-radius: 4px;
            padding: 5px;
            background-color: #f0f0f0;
        }
        QPushButton:hover {
            background-color: #e0e0e0;
        }
        QPushButton:pressed {
            background-color: #d0d0d0;
        }
        QCheckBox {
            spacing: 5px;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border: 1px solid #999999;
            border-radius: 3px;
        }
        QCheckBox::indicator:checked {
            background-color: #0078d7;
            border: 1px solid #0078d7;
        }
        QCheckBox::indicator:unchecked {
            background-color: white;
        }
    """)

def main():
    app = QApplication([])
    set_light_theme(app)
    window = MainWindow()
    window.show()
    app.exec()

if __name__ == "__main__":
    main() 