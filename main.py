"""
LiveLLM - Local Voice Assistant Pipeline
Microphone -> faster-whisper STT -> Ollama LLM -> Piper TTS -> Speaker

Usage: python main.py
Requirements: Ollama running with qwen2.5:7b pulled
"""

import sounddevice as sd
import numpy as np
import threading
import queue
import time
import sys

import ollama
from faster_whisper import WhisperModel
from piper import PiperVoice
from piper.config import SynthesisConfig


# --- Configuration ---
OLLAMA_MODEL = "qwen2.5:7b"
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.1  # seconds per audio block
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium
SILENCE_DURATION = 1.5  # seconds of silence = end of utterance
MIN_SPEECH_DURATION = 0.5  # minimum speech length to process
DEBUG_LEVELS = True  # show live audio RMS levels
# --- Piper Voice Config ---
# Any path in models/piper/ will also appear in the launch-time voice picker.
PIPER_VOICE_PATH = "models/piper/en_US-amy-medium.onnx"
TTS_SPEED = 1.2          # 1.0 natural; >1 faster, <1 slower (length_scale = 1/TTS_SPEED)
TTS_NOISE_SCALE = 0.85   # Prosody/intonation variability. None = voice default (~0.667).
                         #   lower -> flatter/monotone; higher -> more expressive pitch swings
TTS_NOISE_W_SCALE = 1.0  # Rhythm/timing variability. None = voice default (~0.8).
                         #   lower -> metronomic pacing; higher -> looser, more casual rhythm
TTS_VOLUME = 1.0         # Output gain multiplier (1.0 = unchanged)
TTS_NORMALIZE = True     # Normalize loudness across sentences. False preserves natural dynamics.
TTS_SPEAKER_ID = None    # Only for multi-speaker voices (e.g. en_US-libritts_r has ~900 ids)
INPUT_DEVICE = None  # Set to a device index to override (see list_devices.py)
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise and "
    "conversational - aim for 1-3 sentences unless the user asks for detail."
)

# --- Shared State ---
audio_queue = queue.Queue()
tts_queue = queue.Queue()
is_speaking = threading.Event()
conversation_history = []


def synthesize_piper(voice, text):
    """Run Piper on one sentence. Returns (int16_samples, sample_rate)."""
    syn_config = SynthesisConfig(
        length_scale=1.0 / TTS_SPEED,
        noise_scale=TTS_NOISE_SCALE,
        noise_w_scale=TTS_NOISE_W_SCALE,
        volume=TTS_VOLUME,
        normalize_audio=TTS_NORMALIZE,
        speaker_id=TTS_SPEAKER_ID,
    )
    chunks = [c.audio_int16_array for c in voice.synthesize(text, syn_config=syn_config)]
    samples = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int16)
    sample_rate = voice.config.sample_rate
    # Pad ~200ms trailing silence so sounddevice doesn't clip the sentence tail.
    if samples.size:
        samples = np.concatenate([samples, np.zeros(int(sample_rate * 0.2), dtype=np.int16)])
    return samples, sample_rate


def tts_worker():
    # Using Piper via the piper-tts pip package (Try A: pip install succeeded).
    try:
        voice = PiperVoice.load(PIPER_VOICE_PATH)
        print("[TTS ready]", file=sys.stderr)
    except Exception as e:
        print(f"[TTS init failed: {e}]", file=sys.stderr)
        while True:
            if tts_queue.get() is None:
                break
        return

    while True:
        text = tts_queue.get()
        if text is None:
            break
        is_speaking.set()
        try:
            samples, sample_rate = synthesize_piper(voice, text)
            if samples.size:
                sd.play(samples, sample_rate)
                sd.wait()
        except Exception as e:
            print(f"[TTS error: {e}]", file=sys.stderr)
        time.sleep(0.05)
        if tts_queue.empty():
            is_speaking.clear()


def flush_sentences(buffer):
    """Split buffer at sentence boundaries. Returns (sentences_list, remaining_buffer)."""
    sentences = []
    delimiters = [". ", "! ", "? ", ".\n", "!\n", "?\n"]

    changed = True
    while changed:
        changed = False
        best_pos = len(buffer)
        best_len = 0
        for d in delimiters:
            pos = buffer.find(d)
            if pos != -1 and pos < best_pos:
                best_pos = pos
                best_len = len(d)
                changed = True
        if changed:
            sentence = buffer[: best_pos + best_len].strip()
            buffer = buffer[best_pos + best_len :]
            if sentence:
                sentences.append(sentence)

    return sentences, buffer


def process_with_llm(text):
    """Send transcribed text to Ollama, stream response, queue TTS."""
    sys.stdout.write("\r" + " " * 80 + "\r")
    print(f"\n You: {text}")
    print(" Assistant: ", end="", flush=True)

    conversation_history.append({"role": "user", "content": text})

    sentence_buffer = ""
    full_response = ""

    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history
        stream = ollama.chat(model=OLLAMA_MODEL, messages=messages, stream=True)

        for chunk in stream:
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response += token
            sentence_buffer += token

            sentences, sentence_buffer = flush_sentences(sentence_buffer)
            for s in sentences:
                tts_queue.put(s)

        # Flush leftover text
        if sentence_buffer.strip():
            tts_queue.put(sentence_buffer.strip())

        conversation_history.append({"role": "assistant", "content": full_response})
        print()

    except Exception as e:
        print(f"\n[LLM error: {e}]")
        print("Make sure Ollama is running (ollama serve).")


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio: {status}]", file=sys.stderr)
    audio_queue.put(np.frombuffer(indata, dtype=np.int16).copy())


def pick_input_device():
    """Let the user pick a mic if INPUT_DEVICE is not set."""
    if INPUT_DEVICE is not None:
        info = sd.query_devices(INPUT_DEVICE)
        print(f"  Using configured device {INPUT_DEVICE}: {info['name']}")
        return INPUT_DEVICE

    # List input devices
    devices = sd.query_devices()
    input_devs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0 and "Windows DirectSound" not in d["name"] \
                and "WDM-KS" not in d["name"] and "Sound Mapper" not in d["name"]:
            input_devs.append((i, d))

    print("  Available microphones:")
    for idx, (i, d) in enumerate(input_devs):
        default_mark = " (default)" if i == sd.default.device[0] else ""
        print(f"    [{idx}] {d['name']}{default_mark}")

    while True:
        try:
            choice = input(f"  Pick a mic [0-{len(input_devs)-1}]: ").strip()
            choice_idx = int(choice)
            if 0 <= choice_idx < len(input_devs):
                dev_id = input_devs[choice_idx][0]
                print(f"  Selected: {input_devs[choice_idx][1]['name']}")
                return dev_id
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


def calibrate_mic(device):
    """Record 2 seconds of ambient noise to auto-set thresholds."""
    print("  Stay quiet for 2 seconds to calibrate mic...", flush=True)
    recording = sd.rec(
        int(2 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16",
        device=device,
    )
    sd.wait()
    noise_rms = np.sqrt(np.mean(recording.astype(np.float64) ** 2))
    start_threshold = max(noise_rms * 1.5, 200)
    continue_threshold = max(noise_rms * 1.2, 150)
    print(f"  Noise floor: {noise_rms:.0f}")
    print(f"  Start threshold: {start_threshold:.0f} | Continue threshold: {continue_threshold:.0f}")
    return start_threshold, continue_threshold


def check_ollama():
    try:
        ollama.list()
        return True
    except Exception:
        return False


def pick_voice():
    """Scan models/piper/ for .onnx voices and let the user pick one."""
    global PIPER_VOICE_PATH
    import glob
    import os

    voices = sorted(glob.glob("models/piper/*.onnx"))
    if not voices:
        print("  No Piper voices found in models/piper/. Run run.bat to download.")
        sys.exit(1)

    if len(voices) == 1:
        PIPER_VOICE_PATH = voices[0]
        print(f"  Using only installed voice: {os.path.basename(PIPER_VOICE_PATH)}")
        return

    default_voice = PIPER_VOICE_PATH.replace("\\", "/")
    print("  Installed Piper voices:")
    for idx, path in enumerate(voices):
        mark = " (default)" if path.replace("\\", "/") == default_voice else ""
        print(f"    [{idx}] {os.path.basename(path)}{mark}")

    while True:
        try:
            choice = input(f"  Pick a voice [0-{len(voices) - 1}]: ").strip()
            if choice == "":
                print(f"  Selected: {os.path.basename(PIPER_VOICE_PATH)}")
                return
            choice_idx = int(choice)
            if 0 <= choice_idx < len(voices):
                PIPER_VOICE_PATH = voices[choice_idx]
                print(f"  Selected: {os.path.basename(PIPER_VOICE_PATH)}")
                return
        except (ValueError, EOFError):
            pass
        print("  Invalid choice, try again.")


def main():
    print("=" * 50)
    print("  LiveLLM - Local Voice Assistant")
    print("=" * 50)

    # Pre-flight checks
    print("\n[1/4] Checking Ollama...", end=" ", flush=True)
    if not check_ollama():
        print("FAILED - run 'ollama serve' first.")
        sys.exit(1)
    print("OK")

    print("[2/5] Loading Whisper STT model...", end=" ", flush=True)
    whisper_model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
    print("OK")

    print("[3/5] Selecting Piper voice...")
    pick_voice()

    print("[4/5] Starting TTS engine...", end=" ", flush=True)
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    print("OK")

    print("[5/5] Selecting microphone...")
    device_id = pick_input_device()

    print("  Calibrating...")
    start_threshold, continue_threshold = calibrate_mic(device_id)

    print("\n Ready! Speak into your microphone.")
    if DEBUG_LEVELS:
        print(" (debug: showing live RMS levels)")
    print(" Press Ctrl+C to exit.\n")

    # VAD state
    audio_buffer = []
    is_speech = False
    silence_blocks = 0
    speech_blocks = 0
    blocks_for_silence = int(SILENCE_DURATION / BLOCK_DURATION)
    min_speech_blocks = int(MIN_SPEECH_DURATION / BLOCK_DURATION)
    level_counter = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            device=device_id,
            callback=audio_callback,
        ):
            while True:
                try:
                    data = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Skip mic input while assistant is speaking
                if is_speaking.is_set():
                    audio_buffer = []
                    is_speech = False
                    silence_blocks = 0
                    speech_blocks = 0
                    continue

                rms = np.sqrt(np.mean(data.astype(np.float64) ** 2))

                # Show live levels every ~0.5s when idle
                if DEBUG_LEVELS and not is_speech:
                    level_counter += 1
                    if level_counter % 5 == 0:
                        bar_len = min(int(rms / 100), 50)
                        marker = ">>>" if rms > start_threshold else "   "
                        sys.stdout.write(
                            f"\r  rms: {rms:>6.0f} |{'#' * bar_len:<50}| "
                            f"thresh: {start_threshold:.0f} {marker}"
                        )
                        sys.stdout.flush()

                # Use start_threshold to begin, continue_threshold to keep going
                active_threshold = continue_threshold if is_speech else start_threshold

                if rms > active_threshold:
                    # Speech detected (or continuing)
                    if not is_speech:
                        is_speech = True
                    silence_blocks = 0
                    speech_blocks += 1
                    audio_buffer.append(data)
                    sys.stdout.write(
                        f"\r  [listening... {speech_blocks * BLOCK_DURATION:.1f}s]"
                        + " " * 50
                    )
                    sys.stdout.flush()
                elif is_speech:
                    # Silence during speech — keep buffering (captures pauses)
                    silence_blocks += 1
                    audio_buffer.append(data)

                    if silence_blocks >= blocks_for_silence:
                        if speech_blocks >= min_speech_blocks:
                            # End of utterance - transcribe
                            full_audio = np.concatenate(audio_buffer)
                            audio_buffer = []
                            is_speech = False
                            silence_blocks = 0
                            speech_blocks = 0

                            sys.stdout.write("\r  [transcribing...]" + " " * 50)
                            sys.stdout.flush()

                            audio_float = full_audio.astype(np.float32) / 32768.0
                            segments, _info = whisper_model.transcribe(
                                audio_float, beam_size=5, language="en"
                            )
                            text = " ".join(
                                s.text for s in segments if s.no_speech_prob < 0.6
                            ).strip()

                            if text and len(text) > 1:
                                process_with_llm(text)
                            else:
                                sys.stdout.write("\r" + " " * 80 + "\r")
                                sys.stdout.flush()
                        else:
                            # Too short, discard
                            audio_buffer = []
                            is_speech = False
                            silence_blocks = 0
                            speech_blocks = 0

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        tts_queue.put(None)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
