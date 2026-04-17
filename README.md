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

On startup you'll be asked to pick a **Piper voice** and a **microphone**, then the app calibrates for background noise. After that, just talk — the assistant will:

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
| `PIPER_VOICE_PATH` | `models/piper/en_US-ryan-high.onnx` | Default voice if only one is installed; also the default highlighted in the launch picker (see [Changing the Voice](#changing-the-voice)) |
| `TTS_SPEED` | `1.2` | Piper speech speed (higher = faster; maps to `1/length_scale`) |
| `TTS_NOISE_SCALE` | `None` | Prosody/intonation variability. `None` = voice default (~0.667). Lower = flatter, higher = more expressive pitch swings |
| `TTS_NOISE_W_SCALE` | `None` | Rhythm/timing variability. `None` = voice default (~0.8). Lower = metronomic, higher = looser pacing |
| `TTS_VOLUME` | `1.0` | Output gain multiplier |
| `TTS_NORMALIZE` | `True` | Normalize loudness across sentences. Set `False` to preserve natural dynamics |
| `TTS_SPEAKER_ID` | `None` | Multi-speaker voices only (e.g. `en_US-libritts_r` has ~900 ids) |
| `INPUT_DEVICE` | `None` | Set to a device index to skip the mic picker |
| `DEBUG_LEVELS` | `True` | Show live audio RMS meter while idle |

Run `python list_devices.py` to see all audio device indices.

## Changing the Voice

`run.bat` downloads a curated set of seven English Piper voices on first launch (~500 MB total). At startup the app lists every `.onnx` in `models/piper/` and lets you pick one for the session:

```
[3/5] Selecting Piper voice...
  Installed Piper voices:
    [0] en_GB-alan-medium.onnx
    [1] en_GB-jenny_dioco-medium.onnx
    [2] en_US-amy-medium.onnx
    [3] en_US-hfc_female-medium.onnx
    [4] en_US-lessac-medium.onnx
    [5] en_US-libritts_r-medium.onnx
    [6] en_US-ryan-high.onnx (default)
  Pick a voice [0-6]:
```

The default highlighted in the picker is whatever `PIPER_VOICE_PATH` points at in `main.py`. To make a different voice the permanent default, edit that constant.

### Bundled voices

| Voice | Size | Character |
|---|---|---|
| `en_US-ryan-high` (default) | ~116 MB | Male, American, broadcast-style |
| `en_US-amy-medium` | ~63 MB | Female, American, warm |
| `en_US-hfc_female-medium` | ~63 MB | Female, American, neutral/clean |
| `en_US-lessac-medium` | ~61 MB | Female, American, audiobook cadence |
| `en_GB-jenny_dioco-medium` | ~63 MB | Female, British |
| `en_GB-alan-medium` | ~63 MB | Male, British |
| `en_US-libritts_r-medium` | ~75 MB | Multi-speaker (~900 speakers — select via `speaker_id`) |

### Adding more voices

Any `.onnx` + matching `.onnx.json` pair dropped into `models/piper/` will show up in the picker automatically. Browse the full catalog — more accents, speakers, quality tiers — at the [Piper voices repo on Hugging Face](https://huggingface.co/rhasspy/piper-voices/tree/main). Path pattern: `<lang>/<locale>/<speaker>/<quality>/<name>.onnx`. Example:

```bash
curl -L -o models/piper/en_US-joe-medium.onnx ^
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx
curl -L -o models/piper/en_US-joe-medium.onnx.json ^
  https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/joe/medium/en_US-joe-medium.onnx.json
```

Restart the app and it'll appear in the picker.

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
