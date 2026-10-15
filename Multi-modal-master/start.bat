@echo off
title TraceLearn_Launcher

echo ============================================================
echo   TraceLearn - starting development server
echo ============================================================

if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment venv
    call venv\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment .venv
    call .venv\Scripts\activate.bat
) else (
    echo [WARN] No virtual environment found. Using the system Python.
    echo        Create one with:  py -m venv venv ^&^& venv\Scripts\activate ^&^& pip install -r requirements.txt
)

echo [INFO] Starting TraceLearn on http://127.0.0.1:5000
start "TraceLearn_Server" cmd /k "title TraceLearn_Server && python run.py"

echo [SUCCESS] Server starting in a separate window titled TraceLearn_Server
echo [URL]     http://127.0.0.1:5000
echo [STOP]    Run stop.bat to shut it down
echo ============================================================
