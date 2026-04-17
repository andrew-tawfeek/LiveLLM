"""Smoke test: synthesize one sentence with Piper and write out.wav.

Shares the synth code path with main.tts_worker via main.synthesize_piper.
Does not play audio.
"""

import os
import sys
import wave

# Run from repo root so the relative PIPER_VOICE_PATH resolves.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from piper import PiperVoice  # noqa: E402

import main  # noqa: E402


def run():
    text = "Hello, this is a test of the Piper neural voice engine."
    voice = PiperVoice.load(main.PIPER_VOICE_PATH)
    samples, sample_rate = main.synthesize_piper(voice, text)

    out_path = os.path.join(ROOT, "out.wav")
    with wave.open(out_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.tobytes())

    byte_count = os.path.getsize(out_path)
    duration_sec = samples.size / float(sample_rate)
    print(f"duration_sec={duration_sec:.3f} sample_rate={sample_rate} bytes={byte_count}")


if __name__ == "__main__":
    run()
