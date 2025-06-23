import sys
import os

# Добавляем корень проекта в sys.path, чтобы pytest мог найти модуль lob_analyzer
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 