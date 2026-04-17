# LiveLLM - Local Voice Assistant

A fully local voice assistant pipeline that runs on your machine. Speak into your microphone, get answers spoken back.

```
Microphone -> faster-whisper (STT) -> Ollama LLM -> Piper (TTS) -> Speaker
```

## Requirements

- **Windows 10/11**
- **Python 3.10+** — [python.org/downloads](https://www.python.org/downloads/)
- **Ollama** — [ollama.com/download](https://ollama.com/download)

## Quick Start

1. Install Ollama and pull a model:
   ```
   ollama pull qwen2.5:7b
   ```

2. Double-click **`run.bat`**

That's it. On first launch it will:
- Create a Python virtual environment
- Install all dependencies
- Start Ollama if it isn't running
- Download the Whisper speech recognition model (~140MB, one-time)

## Manual Setup

If you prefer to set things up yourself:

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Make sure Ollama is running
ollama serve

# Run
python main.py
```

## Usage

On startup you'll be asked to pick your microphone, then the app calibrates for background noise. After that, just talk — the assistant will:

1. Listen until you stop speaking (~1.5s of silence)
2. Transcribe your speech with Whisper
3. Stream the response from the LLM
4. Speak each sentence aloud as it arrives

Press **Ctrl+C** to exit.

## Configuration

Edit the constants at the top of `main.py`:

| Setting | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `qwen2.5:7b` | Any model available via `ollama list` |
| `WHISPER_MODEL` | `base` | Whisper size: `tiny`, `base`, `small`, `medium` |
| `SILENCE_DURATION` | `1.5` | Seconds of silence before processing speech |
| `MIN_SPEECH_DURATION` | `0.5` | Minimum speech length to process (filters noise) |
| `TTS_SPEED` | `1.0` | Piper speech speed (higher = faster) |
| `PIPER_VOICE_PATH` | `models/piper/en_US-lessac-medium.onnx` | Piper ONNX voice model |
| `INPUT_DEVICE` | `None` | Set to a device index to skip the mic picker |
| `DEBUG_LEVELS` | `True` | Show live audio RMS meter while idle |

Run `python list_devices.py` to see all audio device indices.

## Troubleshooting

**No audio detected / wrong microphone**
- Run `python list_devices.py` to see available devices
- Set `INPUT_DEVICE` in `main.py` to the correct device index, or pick the right one at the prompt

**Speech not recognized well**
- Change `WHISPER_MODEL` to `small` or `medium` for better accuracy (uses more RAM/CPU)
- Speak closer to the mic
- Use headphones to prevent feedback

**TTS not speaking**
- Check Windows volume and output device
- TTS uses Piper — confirm `models/piper/en_US-lessac-medium.onnx` and its `.json` exist (run.bat downloads them on first run)

**Ollama errors**
- Make sure Ollama is running: `ollama serve`
- Verify the model is pulled: `ollama list`

## How It Works

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 port of OpenAI Whisper, runs on CPU
- **VAD**: Volume-based with auto-calibration and hysteresis (two thresholds to prevent flickering)
- **LLM**: [Ollama](https://ollama.com/) — streams tokens for low latency
- **TTS**: [Piper](https://github.com/rhasspy/piper) neural TTS via the `piper-tts` Python package (ONNX, CPU, real-time)
- **Sentence chunking**: LLM output is buffered and split at sentence boundaries, so TTS starts speaking before the full response is ready
