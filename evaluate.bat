@echo off
echo ============================================================
echo  Evaluating on held-out TEST split...
echo ============================================================
set PYTHON=venv\Scripts\python.exe
"%PYTHON%" scripts\evaluate.py
pause
