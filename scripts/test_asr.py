import os
import time
from pathlib import Path

import whisper

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "asr"
FFMPEG_DIR = PROJECT_ROOT / "runtime" / "ffmpeg" / "ffmpeg-9.0.2-essentials_build" / "bin"

os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")

AUDIO_PATH = PROJECT_ROOT / "data" / "asr_test.wav"

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Audio file not found: {AUDIO_PATH}\n"
        "Add a short English WAV recording at this path before running the test."
    )

print("Loading Whisper small model...")
load_start = time.perf_counter()
model = whisper.load_model("small", device="cpu", download_root=str(MODEL_DIR))
load_time = time.perf_counter() - load_start
print(f"Model loaded in {load_time:.2f} seconds.")

print("Transcribing audio on CPU...")
transcribe_start = time.perf_counter()
result = model.transcribe(str(AUDIO_PATH), language="en", fp16=False)
transcribe_time = time.perf_counter() - transcribe_start

print("\n--- ASR Results ---")
print(f"Audio file: {AUDIO_PATH}")
print(f"Transcription time: {transcribe_time:.2f} seconds")
print("Recognized text:")
print(result["text"].strip())
