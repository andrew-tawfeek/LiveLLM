"""Kokoro TTS smoke test: synthesize a sentence to out.wav and print stats."""
import os
import sys

# Make the repo root importable so we pick up config constants from main.py
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import soundfile as sf
from kokoro_onnx import Kokoro

from main import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH, TTS_VOICE, TTS_SPEED


def main():
    model_path = os.path.join(REPO_ROOT, KOKORO_MODEL_PATH)
    voices_path = os.path.join(REPO_ROOT, KOKORO_VOICES_PATH)
    kokoro = Kokoro(model_path, voices_path)
    samples, sr = kokoro.create(
        "Hello, this is a test of the Kokoro neural voice engine.",
        voice=TTS_VOICE,
        speed=TTS_SPEED,
        lang="en-us",
    )
    out_path = os.path.join(REPO_ROOT, "out.wav")
    sf.write(out_path, samples, sr)
    duration = len(samples) / float(sr)
    size = os.path.getsize(out_path)
    print(f"duration_sec={duration:.3f} sample_rate={sr} bytes={size}")


if __name__ == "__main__":
    main()
