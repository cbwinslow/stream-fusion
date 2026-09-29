@echo off
REM StreamFusion Windows Daemon Launcher
REM Runs the StreamFusion 24/7 background supervisor using the project's local virtual environment.

cd /d "%~dp0\..\.."

if exist ".venv\Scripts\python.exe" (
    echo Starting StreamFusion Homelab Daemon...
    .venv\Scripts\python.exe -m stream_fusion.cli daemon run --foreground %*
) else (
    echo [ERROR] Virtual environment not found at .venv\Scripts\python.exe
    echo Please initialize your virtual environment first.
    exit /b 1
)
