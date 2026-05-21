@echo off
set PYTHON=venv\Scripts\python.exe
echo Downloading CTU Drowsiness Dataset (6,624 images)...
"%PYTHON%" scripts\download_dataset.py
pause
