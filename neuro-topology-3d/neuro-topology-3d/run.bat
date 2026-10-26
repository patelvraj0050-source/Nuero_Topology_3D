@echo off
title NeuroTopology 3D - Server Launcher
echo ======================================================================
echo           NEUROTOPOLOGY 3D: REAL-TIME INTERPRETABILITY ENGINE
echo ======================================================================
echo.
echo [*] Checking Python runtime...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [!] Standard 'python' command not found in PATH.
    echo [*] Attempting local Python 3.11 path...
    set PY_PATH="%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
) else (
    set PY_PATH=python
)

echo [*] Starting NeuroTopology 3D server on http://127.0.0.1:8000 ...
start "" http://127.0.0.1:8000
%PY_PATH% backend\server.py 8000
pause
