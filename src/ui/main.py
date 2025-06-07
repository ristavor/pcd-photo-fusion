#!/usr/bin/env python3
# src/ui/main.py

import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    """
    Точка входа для запуска GUI-интерфейса калибровки.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
