#!/usr/bin/env python3
# src/main_test.py

import os
import sys

# Добавляем папку src в PYTHONPATH, чтобы можно было импортировать ui
sys.path.insert(0, os.path.dirname(__file__))

from ui.main import main

if __name__ == "__main__":
    main()
