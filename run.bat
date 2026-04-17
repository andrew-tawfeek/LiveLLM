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
    echo [Setup] Syncing dependencies with requirements.txt...
    pip install -q -r requirements.txt
    if errorlevel 1 (
        echo ERROR: Failed to sync dependencies.
        pause
        exit /b 1
    )
)

:: Download Piper voice models on first run (curated English set, ~500 MB total)
if not exist "models\piper" mkdir "models\piper"
call :download_voice en_US/ryan/high       en_US-ryan-high
call :download_voice en_US/amy/medium      en_US-amy-medium
call :download_voice en_US/hfc_female/medium en_US-hfc_female-medium
call :download_voice en_US/lessac/medium   en_US-lessac-medium
call :download_voice en_GB/jenny_dioco/medium en_GB-jenny_dioco-medium
call :download_voice en_GB/alan/medium     en_GB-alan-medium
call :download_voice en_US/libritts_r/medium en_US-libritts_r-medium
goto :voices_done

:download_voice
:: %1 = HF sub-path (e.g. en_US/ryan/high), %2 = voice filename stem
if not exist "models\piper\%~2.onnx" (
    echo [Setup] Downloading Piper voice: %~2 ...
    curl -L -o "models\piper\%~2.onnx" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/%~1/%~2.onnx"
    if errorlevel 1 (
        echo ERROR: Failed to download %~2.onnx
        pause
        exit /b 1
    )
)
if not exist "models\piper\%~2.onnx.json" (
    curl -L -o "models\piper\%~2.onnx.json" "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/%~1/%~2.onnx.json"
    if errorlevel 1 (
        echo ERROR: Failed to download %~2.onnx.json
        pause
        exit /b 1
    )
)
exit /b 0

:voices_done

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
