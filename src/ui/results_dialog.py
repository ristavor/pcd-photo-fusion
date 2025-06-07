# src/ui/results_dialog.py

import json
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton,
    QFileDialog
)


class ResultsDialog(QDialog):
    """
    Диалог показа и сохранения результатов калибровки:
      - отображает матрицу R (3×3) и вектор T (3×1)
      - позволяет сохранить их в JSON или TXT
    """

    def __init__(self, rvec: np.ndarray, tvec: np.ndarray, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Результаты калибровки")
        self._rvec = rvec
        self._tvec = tvec.flatten()
        self._compute_rt_matrices()
        self._init_ui()
        self._connect_signals()

    def _compute_rt_matrices(self):
        # преобразуем rvec → матрицу R 3×3
        R_mat, _ = cv2.Rodrigues(self._rvec)
        self.R = R_mat  # numpy.ndarray shape (3,3)
        self.T = self._tvec  # numpy.ndarray shape (3,)

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Матрица R
        layout.addWidget(QLabel("Матрица R (3×3):"))
        self._r_text = QTextEdit(self)
        self._r_text.setReadOnly(True)
        r_lines = []
        for row in self.R:
            r_lines.append("    ".join(f"{v:.6f}" for v in row))
        self._r_text.setPlainText("\n".join(r_lines))
        layout.addWidget(self._r_text)

        # Вектор T
        layout.addWidget(QLabel("Вектор T (3×1):"))
        self._t_text = QTextEdit(self)
        self._t_text.setReadOnly(True)
        t_line = "    ".join(f"{v:.6f}" for v in self.T)
        self._t_text.setPlainText(t_line)
        layout.addWidget(self._t_text)

        # Кнопки Сохранить и Закрыть
        btn_layout = QHBoxLayout()
        self._save_btn = QPushButton("Сохранить в файл", self)
        self._close_btn = QPushButton("Закрыть", self)
        btn_layout.addWidget(self._save_btn)
        btn_layout.addWidget(self._close_btn)
        layout.addLayout(btn_layout)

    def _connect_signals(self):
        self._save_btn.clicked.connect(self._on_save)
        self._close_btn.clicked.connect(self.accept)

    def _on_save(self):
        path, fmt = QFileDialog.getSaveFileName(
            self,
            "Сохранить результаты",
            "",
            "JSON файлы (*.json);;Текстовые файлы (*.txt)"
        )
        if not path:
            return
        if path.lower().endswith(".json"):
            self._save_json(path)
        else:
            self._save_txt(path)

    def _save_json(self, path: str):
        data = {
            "R": self.R.tolist(),
            "T": self.T.tolist()
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _save_txt(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            # Сохраняем R
            for row in self.R:
                f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
            f.write("\n")
            # Сохраняем T
            f.write(" ".join(f"{v:.6f}" for v in self.T) + "\n")
