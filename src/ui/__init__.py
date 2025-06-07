# src/ui/__init__.py

"""
Пакет ui: GUI для проведения калибровки Camera ↔ LiDAR.

Модули:
  - main: точка входа приложения
  - main_window: главное окно с логом и кнопкой запуска
  - calibration_dialog: диалог ввода параметров калибровки
  - calibration_worker: выполнение пайплайна калибровки в потоке
  - results_dialog: отображение и сохранение результатов R/T
  - utils: вспомогательные функции (диалоги выбора файлов, форматирование R/T и т.д.)
"""

__all__ = [
    "MainWindow",
    "CalibrationDialog",
    "CalibrationWorker",
    "ResultsDialog",
    "choose_image_file",
    "choose_cloud_file",
    "format_rt",
    "rt_to_dict",
    "save_rt_json",
    "save_rt_txt",
    "load_stylesheet",
    "main"
]

# реэкспорт основных классов и функций
from .main_window import MainWindow
from .calibration_dialog import CalibrationDialog
from .calibration_worker import CalibrationWorker
from .results_dialog import ResultsDialog
from .utils import (
    choose_image_file,
    choose_cloud_file,
    format_rt,
    rt_to_dict,
    save_rt_json,
    save_rt_txt,
    load_stylesheet,
)
from .main import main
