@echo off
set PYTHON=venv\Scripts\python.exe
if not exist "models\best.pt" (
    echo ERROR: models\best.pt not found. Run train.bat first.
    pause
    exit /b 1
)
echo Starting In-Cabin Drowsiness Detection System V2...
echo Press Q in the camera window to quit.
"%PYTHON%" detect.py %*
