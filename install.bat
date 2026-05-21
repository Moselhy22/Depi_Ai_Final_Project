@echo off
echo ============================================================
echo  Drowsiness Detection V2 - Dependency Installer
echo  Python 3.11.5 + CUDA 12.1 + YOLO11 (GTX 1660 Super)
echo ============================================================

set PIP=%~dp0venv\Scripts\pip.exe
set PYTHON=%~dp0venv\Scripts\python.exe

echo [1/3] Upgrading pip...
"%PYTHON%" -m pip install --upgrade pip --quiet

echo [2/3] Installing PyTorch 2.x with CUDA 12.1 (using cache)...
"%PIP%" install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo [3/3] Installing remaining dependencies (YOLO11 requires latest ultralytics)...
"%PIP%" install ultralytics>=8.3.0 opencv-python>=4.9.0 mediapipe>=0.10.0 scipy>=1.13.0 numpy>=1.26.0 pygame>=2.5.0 roboflow>=1.1.0 matplotlib>=3.8.0 seaborn>=0.13.0 tqdm>=4.66.0

echo.
echo Verifying installation...
"%PYTHON%" -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
"%PYTHON%" -c "from ultralytics import YOLO; print('ultralytics: OK')"

echo ============================================================
echo  Installation complete!
echo ============================================================
