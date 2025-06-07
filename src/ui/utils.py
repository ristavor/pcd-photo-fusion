# src/ui/utils.py

import json
import cv2
from typing import Tuple, Any, Dict
from PySide6.QtWidgets import QFileDialog, QApplication


def choose_image_file(parent=None) -> str:
    """
    Открывает диалог выбора файла изображения шахматки.
    :param parent: родительское окно
    :return: путь к выбранному файлу или пустая строка
    """
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Выберите изображение шахматки",
        "",
        "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*)"
    )
    return path or ""


def choose_cloud_file(parent=None) -> str:
    """
    Открывает диалог выбора файла облака точек.
    :param parent: родительское окно
    :return: путь к выбранному файлу или пустая строка
    """
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Выберите файл облака точек",
        "",
        "PointCloud (*.pcd *.ply *.bin *.txt);;Все файлы (*)"
    )
    return path or ""


def format_rt(rvec: Any, tvec: Any) -> Tuple[str, str]:
    """
    Форматирует R и T (по аналогии с ResultsDialog) в строковые представления.
    :param rvec: Rodrigues-вектор (3,) или (3,1)
    :param tvec: вектор трансляции (3,) или (3,1)
    :return: кортеж (text_R, text_T)
    """
    # Преобразуем Rodrigues-вектор в матрицу R
    R_mat, _ = cv2.Rodrigues(rvec)
    # Выравнивание
    text_R = "\n".join("    ".join(f"{v:.6f}" for v in row) for row in R_mat)
    t = tvec.flatten()
    text_T = "    ".join(f"{v:.6f}" for v in t)
    return text_R, text_T


def rt_to_dict(rvec: Any, tvec: Any) -> Dict[str, Any]:
    """
    Преобразует R/T в JSON-сериализуемый словарь.
    :param rvec: Rodrigues-вектор (3,) или (3,1)
    :param tvec: вектор трансляции (3,) или (3,1)
    :return: {"R": [[...],...], "T": [...]}
    """
    R_mat, _ = cv2.Rodrigues(rvec)
    return {
        "R": R_mat.tolist(),
        "T": tvec.flatten().tolist()
    }


def save_rt_json(path: str, rvec: Any, tvec: Any) -> None:
    """
    Сохраняет R/T в JSON-файл.
    """
    data = rt_to_dict(rvec, tvec)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def save_rt_txt(path: str, rvec: Any, tvec: Any) -> None:
    """
    Сохраняет R/T в текстовый файл:
      три строки для R, затем пустая строка, затем одна строка для T.
    """
    R_mat, _ = cv2.Rodrigues(rvec)
    t = tvec.flatten()
    with open(path, "w", encoding="utf-8") as f:
        for row in R_mat:
            f.write(" ".join(f"{v:.6f}" for v in row) + "\n")
        f.write("\n")
        f.write(" ".join(f"{v:.6f}" for v in t) + "\n")


def load_stylesheet(path: str) -> str:
    """
    Загружает QSS-стили из файла.
    :param path: путь к файлу .qss
    :return: содержимое файла как строка
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""
