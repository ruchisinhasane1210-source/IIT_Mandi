@echo off
title TraceLearn_Stop

echo ============================================================
echo   TraceLearn - stopping development server
echo ============================================================

for /f "tokens=5" %%p in ('netstat -ano ^| findstr :5000 ^| findstr LISTENING') do (
    echo [INFO] Killing process %%p on port 5000
    taskkill /PID %%p /F >nul 2>&1
)

echo [DONE] Port 5000 is free.
echo ============================================================
