@echo off
setlocal
cd /d "%~dp0"
title Particle Flow Analysis Tool

echo ==========================================
echo   Particle Flow Video Analysis Tool
echo ==========================================
echo.

rem Try the Python launcher first, then fall back to python.exe.
py -3 --version >nul 2>nul
if not errorlevel 1 (
    set "PYTHON=py -3"
    goto :python_found
)

python --version >nul 2>nul
if not errorlevel 1 (
    set "PYTHON=python"
    goto :python_found
)

echo Python was not found on this computer.
echo Please install Python 3.10 or newer, then run this file again.
echo.
pause
exit /b 1

:python_found
if not exist ".venv\Scripts\python.exe" (
    echo First-time setup: creating a private Python environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 goto :setup_error

    echo Installing required packages. This may take a few minutes...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    if errorlevel 1 goto :setup_error
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :setup_error
)

echo Starting Particle Flow Analysis Tool...
echo Your browser should open automatically.
echo Keep this window open while using the tool.
echo.
".venv\Scripts\python.exe" -m streamlit run app.py
exit /b

:setup_error
echo.
echo Setup failed. Please check your internet connection and Python installation.
pause
exit /b 1
