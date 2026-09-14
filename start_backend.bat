@echo off
title Plant Disease Detection - Backend Server
echo ======================================================================
echo   Starting PlantGuard AI Backend Server...
echo ======================================================================
echo.
set KMP_DUPLICATE_LIB_OK=TRUE
python run_server.py --host 127.0.0.1 --port 8000 --reload
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Server encountered an error. Press any key to exit.
    pause >nul
)
