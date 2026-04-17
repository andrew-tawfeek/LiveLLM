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
| `PIPER_VOICE_PATH` | `models/piper/en_US-ryan-high.onnx` | Piper ONNX voice model (see [Changing the Voice](#changing-the-voice)) |
| `INPUT_DEVICE` | `None` | Set to a device index to skip the mic picker |
| `DEBUG_LEVELS` | `True` | Show live audio RMS meter while idle |

Run `python list_devices.py` to see all audio device indices.

## Changing the Voice

The default voice is **`en_US-ryan-high`** (male, American, high-fidelity — ~116 MB). To try another speaker, accent, or fidelity tier:

1. Pick a voice from the [Piper voices repo on Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main/en). The path pattern is `en/<locale>/<speaker>/<quality>/<name>.onnx`.

2. Download both files into `models/piper/`. For example, to grab `en_US-amy-medium`:

   ```bash
   curl -L -o models/piper/en_US-amy-medium.onnx ^
     https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx
   curl -L -o models/piper/en_US-amy-medium.onnx.json ^
     https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json
   ```

3. Point `PIPER_VOICE_PATH` in `main.py` at the new `.onnx` file. Restart the app.

### Good voices to try

| Voice | Size | Character |
|---|---|---|
| `en_US-ryan-high` (default) | ~116 MB | Male, clear, broadcast-style |
| `en_US-amy-medium` | ~63 MB | Female, warm |
| `en_US-hfc_female-medium` | ~63 MB | Female, neutral/clean |
| `en_US-lessac-medium` | ~61 MB | Female, audiobook cadence |
| `en_GB-jenny_dioco-medium` | ~63 MB | British female |
| `en_GB-alan-medium` | ~63 MB | British male |
| `en_US-libritts_r-medium` | ~75 MB | Multi-speaker (~900 speakers — select via `speaker_id`) |

### Quality tiers

Every voice exists in multiple sizes. Bigger = richer, slightly slower on CPU:

| Tier | Size | Notes |
|---|---|---|
| `x_low` | ~10–15 MB | noticeable artifacts |
| `low` | ~20–30 MB | usable, thin |
| `medium` | ~60–75 MB | sweet spot |
| `high` | ~110–125 MB | richest — current default |

### Speed and expressiveness

- **`TTS_SPEED`** in `main.py` — `1.0` is natural; `1.1`–`1.2` feels snappier, `0.9` slower.
- For more expressiveness you can pass `noise_scale` and `noise_w_scale` into `SynthesisConfig` inside `synthesize_piper()` (higher values = more prosody variation). Defaults (~0.667 and ~0.8) are usually fine.

### Multi-speaker voices

Models like `en_US-libritts_r-medium` ship hundreds of speakers in one file. Pick one by setting `speaker_id` (an int) on the `SynthesisConfig` in `synthesize_piper()`. See the voice's `.onnx.json` for the speaker list.

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
- TTS uses Piper — confirm `models/piper/en_US-ryan-high.onnx` and its `.json` exist (run.bat downloads them on first run)

**Ollama errors**
- Make sure Ollama is running: `ollama serve`
- Verify the model is pulled: `ollama list`

## How It Works

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) — CTranslate2 port of OpenAI Whisper, runs on CPU
- **VAD**: Volume-based with auto-calibration and hysteresis (two thresholds to prevent flickering)
- **LLM**: [Ollama](https://ollama.com/) — streams tokens for low latency
- **TTS**: [Piper](https://github.com/rhasspy/piper) neural TTS via the `piper-tts` Python package (ONNX, CPU, real-time)
- **Sentence chunking**: LLM output is buffered and split at sentence boundaries, so TTS starts speaking before the full response is ready
