# src/ui/calibration_dialog.py

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog, QLabel, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QHBoxLayout, QVBoxLayout
)


class CalibrationDialog(QDialog):
    """
    Диалог для ввода параметров калибровки:
      - физический размер клетки шахматки
      - количество внутренних углов по X (cols) и Y (rows)
      - файл изображения
      - файл облака точек
    Излучает сигнал startCalibration при нажатии Запустить.
    """
    # Аргументы: square_size (м), cols (int), rows (int), image_path (str), cloud_path (str)
    startCalibration = Signal(float, int, int, str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Параметры калибровки")
        self._init_ui()
        self._connect_signals()
        self._update_run_button()

    def _init_ui(self):
        # Поле для размера клетки
        self.square_size_spin = QDoubleSpinBox(self)
        self.square_size_spin.setRange(0.001, 10.0)
        self.square_size_spin.setDecimals(3)
        self.square_size_spin.setSingleStep(0.001)
        self.square_size_spin.setValue(0.10)

        # Поля для числа внутренних углов
        self.cols_spin = QSpinBox(self)
        self.cols_spin.setRange(1, 100)
        self.cols_spin.setValue(7)

        self.rows_spin = QSpinBox(self)
        self.rows_spin.setRange(1, 100)
        self.rows_spin.setValue(5)

        # Выбор файла изображения
        self.image_line = QLineEdit(self)
        self.image_line.setReadOnly(True)
        self.image_button = QPushButton("Выбрать...", self)

        # Выбор файла облака точек
        self.cloud_line = QLineEdit(self)
        self.cloud_line.setReadOnly(True)
        self.cloud_button = QPushButton("Выбрать...", self)

        # Кнопка запуска
        self.run_button = QPushButton("Запустить", self)
        self.run_button.setEnabled(False)

        # Сборка формы
        form = QFormLayout()
        form.addRow("Размер клетки (м):", self.square_size_spin)

        hl = QHBoxLayout()
        hl.addWidget(self.cols_spin)
        hl.addWidget(QLabel("×"))
        hl.addWidget(self.rows_spin)
        form.addRow("Внутренние углы (cols × rows):", hl)

        img_hl = QHBoxLayout()
        img_hl.addWidget(self.image_line)
        img_hl.addWidget(self.image_button)
        form.addRow("Файл изображения:", img_hl)

        cloud_hl = QHBoxLayout()
        cloud_hl.addWidget(self.cloud_line)
        cloud_hl.addWidget(self.cloud_button)
        form.addRow("Файл облака точек:", cloud_hl)

        # Основной лэйаут
        vbox = QVBoxLayout()
        vbox.addLayout(form)
        vbox.addWidget(self.run_button)
        self.setLayout(vbox)

    def _connect_signals(self):
        self.square_size_spin.valueChanged.connect(self._update_run_button)
        self.cols_spin.valueChanged.connect(self._update_run_button)
        self.rows_spin.valueChanged.connect(self._update_run_button)
        self.image_button.clicked.connect(self._choose_image)
        self.cloud_button.clicked.connect(self._choose_cloud)
        self.run_button.clicked.connect(self._on_run)

    def _choose_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите изображение шахматки",
            "",
            "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"
        )
        if path:
            self.image_line.setText(path)
        self._update_run_button()

    def _choose_cloud(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл облака точек",
            "",
            "PointCloud (*.pcd *.ply *.bin *.txt);;Все файлы (*)"
        )
        if path:
            self.cloud_line.setText(path)
        self._update_run_button()

    def _update_run_button(self):
        """
        Кнопка Запустить активна только если заданы:
          - размер клетки > 0
          - cols >= 1 и rows >= 1
          - непустые пути к файлам
        """
        ok = (
            self.square_size_spin.value() > 0
            and self.cols_spin.value() > 0
            and self.rows_spin.value() > 0
            and bool(self.image_line.text())
            and bool(self.cloud_line.text())
        )
        self.run_button.setEnabled(ok)

    def _on_run(self):
        """
        Сбор параметров и запуск калибровки.
        Эмиттит startCalibration и закрывает диалог.
        """
        square_size = float(self.square_size_spin.value())
        cols = int(self.cols_spin.value())
        rows = int(self.rows_spin.value())
        image_path = self.image_line.text()
        cloud_path = self.cloud_line.text()

        # Сигнал для внешнего контроллера
        self.startCalibration.emit(square_size, cols, rows, image_path, cloud_path)
        self.accept()
