@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install --upgrade pip
pip install loguru pandas numpy pyarrow pytest matplotlib scikit-learn
