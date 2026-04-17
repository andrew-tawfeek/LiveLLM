@echo off
title LiveLLM - Voice Assistant
cd /d "%~dp0"

:: Find Python
where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org/downloads
    pause
    exit /b 1
)

:: Find Ollama (check PATH, then common install locations)
set "OLLAMA_CMD=ollama"
where ollama >nul 2>&1
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
        set "OLLAMA_CMD=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    ) else if exist "%ProgramFiles%\Ollama\ollama.exe" (
        set "OLLAMA_CMD=%ProgramFiles%\Ollama\ollama.exe"
    ) else (
        echo ERROR: Ollama not found. Install from https://ollama.com/download
        pause
        exit /b 1
    )
)

:: Create venv if needed
if not exist "venv\Scripts\activate.bat" (
    echo [Setup] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    call venv\Scripts\activate.bat
    echo [Setup] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies.
        pause
        exit /b 1
    )
) else (
    call venv\Scripts\activate.bat
)

:: Download Piper voice model on first run
if not exist "models\piper\en_US-lessac-medium.onnx" (
    echo [Setup] Downloading Piper voice model...
    if not exist "models\piper" mkdir "models\piper"
    curl -L -o "models\piper\en_US-lessac-medium.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    if errorlevel 1 (
        echo ERROR: Failed to download Piper voice model.
        pause
        exit /b 1
    )
)
if not exist "models\piper\en_US-lessac-medium.onnx.json" (
    echo [Setup] Downloading Piper voice config...
    curl -L -o "models\piper\en_US-lessac-medium.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    if errorlevel 1 (
        echo ERROR: Failed to download Piper voice config.
        pause
        exit /b 1
    )
)

:: Start Ollama if not running
"%OLLAMA_CMD%" list >nul 2>&1
if errorlevel 1 (
    echo [Setup] Starting Ollama...
    start /min "" "%OLLAMA_CMD%" serve
    timeout /t 3 /nobreak >nul
)

:: Pull model if not present
"%OLLAMA_CMD%" list 2>nul | findstr /i "qwen2.5:7b" >nul
if errorlevel 1 (
    echo [Setup] Pulling qwen2.5:7b model ^(this may take a while^)...
    "%OLLAMA_CMD%" pull qwen2.5:7b
)

echo.
python main.py
pause
