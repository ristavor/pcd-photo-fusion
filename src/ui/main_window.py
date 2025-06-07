# src/ui/main_window.py

import sys
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QPushButton, QTextEdit, QVBoxLayout,
    QMessageBox, QApplication
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor

from ui.calibration_dialog import CalibrationDialog
from ui.calibration_worker import CalibrationWorker
from ui.results_dialog import ResultsDialog


class MainWindow(QMainWindow):
    """
    Главное окно приложения. Содержит кнопку для запуска
    процесса калибровки и область логов.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Калибровка Camera ↔ LiDAR")
        self._worker = None
        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        # Центральный виджет
        central = QWidget(self)
        self.setCentralWidget(central)

        # Кнопка запуска
        self._btn_calibrate = QPushButton("Провести калибровку", self)
        self._btn_calibrate.setFixedHeight(40)

        # Область логов
        self._log = QTextEdit(self)
        self._log.setReadOnly(True)
        self._log.setAcceptRichText(False)

        # Вертикальный лэйаут
        layout = QVBoxLayout(central)
        layout.addWidget(self._btn_calibrate)
        layout.addWidget(self._log)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

    def _connect_signals(self):
        self._btn_calibrate.clicked.connect(self._on_calibrate_clicked)

    def _on_calibrate_clicked(self):
        """
        Открывает диалог ввода параметров калибровки.
        """
        dlg = CalibrationDialog(self)
        dlg.startCalibration.connect(self._start_calibration)
        dlg.exec()

    def _start_calibration(self, square_size, cols, rows, image_path, cloud_path):
        """
        Принимает параметры из CalibrationDialog, подготавливает
        и запускает CalibrationWorker.
        """
        # Очищаем логи и блокируем кнопку
        self._log.clear()
        self._btn_calibrate.setEnabled(False)
        self._append_log("Запуск калибровки…")

        # Создаем и настраиваем worker
        self._worker = CalibrationWorker(
            square_size=square_size,
            cols=cols,
            rows=rows,
            image_path=image_path,
            cloud_path=cloud_path
        )
        self._worker.progress.connect(self._append_log)
        self._worker.error.connect(self._handle_error)
        self._worker.finished.connect(self._handle_finished)
        self._worker.start()

    def _append_log(self, message: str):
        """
        Добавляет строку в лог (сдвиг к концу и авто-скролл).
        """
        self._log.append(message)
        cursor = self._log.textCursor()
        cursor.movePosition(QTextCursor.End)
        self._log.setTextCursor(cursor)

    def _handle_error(self, error_message: str):
        """
        Обрабатывает ошибку из worker: показывает MessageBox,
        логирует и разблокирует кнопку.
        """
        self._append_log(f"Ошибка: {error_message}")
        QMessageBox.critical(self, "Ошибка калибровки", error_message)
        self._btn_calibrate.setEnabled(True)

    def _handle_finished(self, rvec, tvec):
        """
        Обрабатывает успешное завершение: логирует, показывает результаты
        и разблокирует кнопку.
        """
        self._append_log("Калибровка успешно завершена.")
        # Показать окно с результатами
        dlg = ResultsDialog(rvec, tvec, parent=self)
        dlg.exec()
        self._btn_calibrate.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.resize(600, 400)
    win.show()
    sys.exit(app.exec())
