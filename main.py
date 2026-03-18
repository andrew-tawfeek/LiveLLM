"""
LiveLLM - Local Voice Assistant Pipeline
Microphone -> Vosk STT -> Ollama LLM -> pyttsx3 TTS -> Speaker

Usage: python main.py
Requirements: Ollama running with qwen2.5:7b pulled
"""

import vosk
import sounddevice as sd
import json
import queue
import threading
import subprocess
import sys
import time
import os
import tempfile

import ollama
import numpy as np


# --- Configuration ---
OLLAMA_MODEL = "qwen2.5:7b"
SAMPLE_RATE = 16000
BLOCK_SIZE = 4000
VOSK_MODEL = "vosk-model-en-us-0.22"  # Larger model for much better accuracy
TTS_RATE = 175
SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep responses concise and "
    "conversational - aim for 1-3 sentences unless the user asks for detail."
)

# --- Shared State ---
audio_queue = queue.Queue()
tts_queue = queue.Queue()
is_speaking = threading.Event()
conversation_history = []


def tts_worker():
    """Background thread: pulls sentences from tts_queue and speaks them via pyttsx3.

    pyttsx3 is initialized inside this thread to avoid COM threading issues on Windows.
    Each sentence is spoken synchronously so we can track when speech finishes.
    """
    import pyttsx3

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)

        voices = engine.getProperty("voices")
        for v in voices:
            if "zira" in v.name.lower() or "david" in v.name.lower():
                engine.setProperty("voice", v.id)
                break

        # Quick test to confirm TTS works
        engine.say(" ")
        engine.runAndWait()
        print("[TTS engine initialized]", file=sys.stderr)
    except Exception as e:
        print(f"\n[TTS init failed: {e}]", file=sys.stderr)
        print("[Falling back to no TTS - text only]", file=sys.stderr)
        # Drain queue without speaking
        while True:
            text = tts_queue.get()
            if text is None:
                break
        return

    while True:
        text = tts_queue.get()
        if text is None:
            break
        is_speaking.set()
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"\n[TTS error: {e}]", file=sys.stderr)
            # Reinitialize engine on failure
            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", TTS_RATE)
            except Exception:
                pass
        # Brief pause to check if more queued speech follows
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
    """Send transcribed text to Ollama, stream response, and queue TTS."""
    clear_line()
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


def clear_line():
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[audio: {status}]", file=sys.stderr)
    audio_queue.put(bytes(indata))


def check_ollama():
    """Verify Ollama is reachable."""
    try:
        ollama.list()
        return True
    except Exception:
        return False


def main():
    print("=" * 50)
    print("  LiveLLM - Local Voice Assistant")
    print("=" * 50)

    # Pre-flight checks
    print("\n[1/3] Checking Ollama...", end=" ", flush=True)
    if not check_ollama():
        print("FAILED")
        print("  Could not connect to Ollama. Run 'ollama serve' first.")
        sys.exit(1)
    print("OK")

    print("[2/3] Loading speech recognition model...", end=" ", flush=True)
    vosk.SetLogLevel(-1)
    model = vosk.Model(model_name=VOSK_MODEL)
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)
    print("OK")

    print("[3/3] Starting TTS engine...", end=" ", flush=True)
    tts_thread = threading.Thread(target=tts_worker, daemon=True)
    tts_thread.start()
    print("OK")

    print("\n Ready! Speak into your microphone.")
    print(" Tip: Use headphones to avoid audio feedback.")
    print(" Press Ctrl+C to exit.\n")

    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="int16",
            channels=1,
            callback=audio_callback,
        ):
            while True:
                data = audio_queue.get()

                # Ignore mic input while the assistant is speaking
                if is_speaking.is_set():
                    continue

                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text and len(text) > 1:
                        process_with_llm(text)
                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text:
                        sys.stdout.write(f"\r  hearing: {partial_text:<70}")
                        sys.stdout.flush()

    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        tts_queue.put(None)
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your microphone is connected and accessible.")
        sys.exit(1)


if __name__ == "__main__":
    main()
