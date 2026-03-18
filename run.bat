@echo off
title LiveLLM - Voice Assistant
cd /d "%~dp0"

:: Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo [Setup] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [Setup] Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

:: Check if Ollama is reachable, start it if not
ollama list >nul 2>&1
if errorlevel 1 (
    echo [Setup] Starting Ollama...
    start /min "" ollama serve
    timeout /t 3 /nobreak >nul
)

:: Check if model is pulled
ollama list 2>nul | findstr /i "qwen2.5:7b" >nul
if errorlevel 1 (
    echo [Setup] Pulling qwen2.5:7b model (this may take a while)...
    ollama pull qwen2.5:7b
)

echo.
python main.py
pause
