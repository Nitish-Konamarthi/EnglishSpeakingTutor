import time
from pathlib import Path

import soundfile as sf
import torch
from kokoro import KPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_PATH = OUTPUT_DIR / "tts_test.wav"

TEXT = (
    "Hello! My name is Nitish. I am building a local English speaking tutor. "
    "I want to improve my fluency, vocabulary, and confidence. "
    "Let's practice speaking English naturally, one conversation at a time."
)

VOICE = "af_heart"
SAMPLE_RATE = 24000

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Device: CPU")
print(f"Voice: {VOICE}")
print(f"Text length: {len(TEXT)} characters")

# Load and initialize Kokoro.
print("\nLoading Kokoro...")
load_start = time.perf_counter()

pipeline = KPipeline(lang_code="a")

load_time = time.perf_counter() - load_start
print(f"Pipeline initialization time: {load_time:.2f} seconds.")

# Generate speech.
print("\nGenerating speech...")
generation_start = time.perf_counter()

audio_chunks = []

with torch.inference_mode():
    for _, _, audio in pipeline(
        TEXT,
        voice=VOICE,
        speed=1.0,
        split_pattern=r"\n+",
    ):
        audio_chunks.append(audio.cpu().numpy())

generation_time = time.perf_counter() - generation_start

if not audio_chunks:
    raise RuntimeError("Kokoro did not generate any audio chunks.")

import numpy as np

audio = np.concatenate(audio_chunks)
sf.write(str(OUTPUT_PATH), audio, SAMPLE_RATE)

audio_duration = len(audio) / SAMPLE_RATE
rtf = generation_time / audio_duration if audio_duration > 0 else float("inf")

print("\n--- TTS Benchmark Results ---")
print(f"Pipeline initialization: {load_time:.2f} seconds")
print(f"Speech generation time: {generation_time:.2f} seconds")
print(f"Generated audio duration: {audio_duration:.2f} seconds")
print(f"Real-time factor (RTF): {rtf:.3f}")
print(f"Output file: {OUTPUT_PATH}")
print(f"Audio file size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")
