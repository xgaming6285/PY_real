@echo off
echo Starting ID Document Verification System...

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.7-3.10 from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Run the application
python run.py

pause 