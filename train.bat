@echo off
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHON=%~dp0venv\Scripts\python.exe
echo ============================================================
echo  YOLO11 Training - In-Cabin Drowsiness Detection V2
echo  300 Epochs ^| BCE Loss ^| GTX 1660 Super
echo ============================================================
"%PYTHON%" "%~dp0scripts\train.py"
pause
