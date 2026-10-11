import os
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import whisper
from kokoro import KPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent

AUDIO_INPUT = PROJECT_ROOT / "data" / "asr_test.wav"
AUDIO_OUTPUT = PROJECT_ROOT / "data" / "pipeline_response.wav"
ASR_MODEL_DIR = PROJECT_ROOT / "models" / "asr"

FFMPEG_DIR = (
    PROJECT_ROOT
    / "runtime"
    / "ffmpeg"
    / "ffmpeg-9.0.2-essentials_build"
    / "bin"
)

SERVER_URL = "http://127.0.0.1:8080"
TTS_VOICE = "af_heart"
SAMPLE_RATE = 24000

os.environ["PATH"] = (
    str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
)

if not AUDIO_INPUT.exists():
    raise FileNotFoundError(f"Input audio not found: {AUDIO_INPUT}")

print("=" * 60)
print("ENGLISH SPEAKING TUTOR - VOICE PIPELINE")
print("=" * 60)

# Stage 1: Speech recognition
print("\n[1/3] Loading Whisper small...")
start = time.perf_counter()

asr_model = whisper.load_model(
    "small",
    device="cpu",
    download_root=str(ASR_MODEL_DIR),
)

asr_load_time = time.perf_counter() - start
print(f"Whisper initialization: {asr_load_time:.2f}s")

print("Transcribing input audio...")
start = time.perf_counter()

transcription = asr_model.transcribe(
    str(AUDIO_INPUT),
    language="en",
    fp16=False,
)

asr_time = time.perf_counter() - start
user_text = transcription["text"].strip()

if not user_text:
    raise RuntimeError("Whisper returned an empty transcription.")

print(f"You said: {user_text}")
print(f"ASR transcription time: {asr_time:.2f}s")

# Stage 2: Local Qwen inference
print("\n[2/3] Sending transcript to local Qwen...")
start = time.perf_counter()

models_response = requests.get(
    f"{SERVER_URL}/v1/models",
    timeout=10,
)
models_response.raise_for_status()

models_data = models_response.json()
model_id = models_data["data"][0]["id"]

prompt = (
    "You are a friendly, patient English-speaking tutor helping an "
    "English learner improve grammar, vocabulary, fluency, and confidence.\n\n"

    "Follow these rules:\n"
    "1. Understand the learner's intended meaning before correcting them.\n"
    "2. If there is a meaningful grammar or word-choice mistake, show "
    "the corrected sentence and briefly explain the main correction.\n"
    "3. Do not invent mistakes. If the sentence is already correct, "
    "say so briefly and naturally.\n"
    "4. Preserve the learner's intended meaning and use natural, "
    "everyday English.\n"
    "5. Avoid correcting harmless differences in speaking style.\n"
    "6. Keep feedback encouraging and easy to understand.\n"
    "7. Continue the conversation with one relevant question that "
    "encourages the learner to speak more.\n"
    "8. Keep your response concise, ideally three to five sentences.\n"
    "9. Do not overwhelm the learner with multiple grammar lessons at once.\n\n"

    f"Learner's message: {user_text}"
)

response = requests.post(
    f"{SERVER_URL}/v1/chat/completions",
    json={
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a supportive English conversation tutor. "
                    "Give accurate, concise corrections when needed, explain "
                    "them simply, and keep the conversation flowing naturally."
                ),
            },
            {
            "role": "user",
            "content": prompt,
            },
        ],
        "temperature": 0.7,
        "max_tokens": 160,
    },
    timeout=180,
)

response.raise_for_status()
response_data = response.json()
assistant_text = response_data["choices"][0]["message"]["content"].strip()

if not assistant_text:
    raise RuntimeError("Qwen returned an empty response.")

llm_time = time.perf_counter() - start

print(f"Learner transcript: {user_text}")
print(f"Tutor response: {assistant_text}")
print(f"Qwen response time: {llm_time:.2f}s")

# Stage 3: Text-to-speech
print("\n[3/3] Initializing Kokoro...")
start = time.perf_counter()

tts_pipeline = KPipeline(lang_code="a")

tts_load_time = time.perf_counter() - start
print(f"Kokoro initialization: {tts_load_time:.2f}s")

print("Generating spoken response...")
start = time.perf_counter()

audio_chunks = []

with torch.inference_mode():
    for _, _, audio in tts_pipeline(
        assistant_text,
        voice=TTS_VOICE,
        speed=1.0,
        split_pattern=r"\n+",
    ):
        audio_chunks.append(audio.cpu().numpy())

tts_time = time.perf_counter() - start

if not audio_chunks:
    raise RuntimeError("Kokoro did not generate any audio.")

audio = np.concatenate(audio_chunks)

sf.write(str(AUDIO_OUTPUT), audio, SAMPLE_RATE)

audio_duration = len(audio) / SAMPLE_RATE
rtf = tts_time / audio_duration if audio_duration else float("inf")

# Summary
print("\n" + "=" * 60)
print("VOICE PIPELINE RESULTS")
print("=" * 60)
print(f"Recognized text: {user_text}")
print(f"Tutor reply: {assistant_text}")
print("\nTiming:")
print(f"Whisper initialization: {asr_load_time:.2f}s")
print(f"Whisper transcription:  {asr_time:.2f}s")
print(f"Qwen inference:         {llm_time:.2f}s")
print(f"Kokoro initialization:  {tts_load_time:.2f}s")
print(f"Kokoro generation:      {tts_time:.2f}s")
print(f"Generated audio length: {audio_duration:.2f}s")
print(f"TTS real-time factor:   {rtf:.3f}")
print(f"Output audio:           {AUDIO_OUTPUT}")

print("\nFirst-response elapsed time, including model initialization: "
      f"{asr_load_time + asr_time + llm_time + tts_load_time + tts_time:.2f}s")

print("Subsequent-response estimate, excluding initialization: "
      f"{asr_time + llm_time + tts_time:.2f}s")

print("\nPipeline completed successfully.")
