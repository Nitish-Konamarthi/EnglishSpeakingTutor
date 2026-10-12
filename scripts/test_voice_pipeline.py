import os
import time
import sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf
import torch
import whisper
from kokoro import KPipeline

if os.name == "nt":
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")


# ============================================================
# Configuration
# ============================================================

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

# Maximum number of user/assistant messages retained.
MAX_HISTORY_MESSAGES = 25

# In-memory conversation history.
# The system prompt is supplied separately and does not count
# toward the 25-message limit.
conversation_history = []

SYSTEM_PROMPT = """
You are a friendly, patient English-speaking tutor helping a learner
improve grammar, vocabulary, fluency, and confidence.

Follow these rules:

1. Understand the learner's intended meaning before correcting them.
2. If there is a meaningful grammar or word-choice mistake, provide
   the corrected sentence and briefly explain the main correction.
3. Do not invent mistakes. If the sentence is already correct,
   say so naturally.
4. Preserve the learner's intended meaning.
5. Use natural, everyday English.
6. Avoid correcting harmless differences in speaking style.
7. Give encouraging and accurate feedback.
8. Remember relevant details from earlier conversation turns.
9. Ask one relevant follow-up question when appropriate.
10. Keep responses concise, ideally 2–4 short sentences.
11. Avoid lengthy lists, unnecessary repetition, and multiple
    grammar lessons at once.
12. Make responses sound natural when spoken aloud.
"""


# ============================================================
# Conversation memory
# ============================================================

def trim_history(history, max_messages=MAX_HISTORY_MESSAGES):
    """
    Keep the latest conversation messages while preserving
    complete user/assistant turns.

    The history contains user and assistant messages only.
    The system prompt is added separately to each API request.
    """
    history = history.copy()

    while len(history) > max_messages:
        # Remove the oldest user message.
        if history and history[0]["role"] == "user":
            history.pop(0)

        # Remove its corresponding assistant response.
        if history and history[0]["role"] == "assistant":
            history.pop(0)

    return history


def generate_tutor_response(model_id, user_text):
    """
    Send the current user message and conversation history
    to the local Qwen model.

    Save both the user message and assistant response.
    """
    global conversation_history

    # Add the current user message.
    conversation_history.append(
        {
            "role": "user",
            "content": user_text,
        }
    )

    # Trim before sending the request.
    conversation_history = trim_history(conversation_history)

    # The system prompt is not part of the stored history.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        *conversation_history,
    ]

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": model_id,
            "messages": messages,
            "temperature": 0.4,
            "max_tokens": 180,
        },
        timeout=180,
    )

    response.raise_for_status()

    response_data = response.json()

    assistant_text = (
        response_data["choices"][0]["message"]["content"].strip()
    )

    if not assistant_text:
        raise RuntimeError("Qwen returned an empty response.")

    # Store the tutor's response for subsequent turns.
    conversation_history.append(
        {
            "role": "assistant",
            "content": assistant_text,
        }
    )

    # Enforce the history limit again.
    conversation_history = trim_history(conversation_history)

    return assistant_text


# ============================================================
# Environment setup
# ============================================================

os.environ["PATH"] = (
    str(FFMPEG_DIR)
    + os.pathsep
    + os.environ.get("PATH", "")
)

if not AUDIO_INPUT.exists():
    raise FileNotFoundError(
        f"Input audio not found: {AUDIO_INPUT}"
    )

print("=" * 60)
print("ENGLISH SPEAKING TUTOR - VOICE PIPELINE")
print("WITH CONVERSATION MEMORY")
print("=" * 60)


# ============================================================
# Stage 1: Speech recognition
# ============================================================

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
    raise RuntimeError(
        "Whisper returned an empty transcription."
    )

print(f"You said: {user_text}")
print(f"ASR transcription time: {asr_time:.2f}s")


# ============================================================
# Stage 2: Local Qwen inference with conversation memory
# ============================================================

print("\n[2/3] Sending transcript and conversation history to Qwen...")

start = time.perf_counter()

# Discover the model served by llama.cpp.
models_response = requests.get(
    f"{SERVER_URL}/v1/models",
    timeout=10,
)

models_response.raise_for_status()

models_data = models_response.json()
available_models = models_data.get("data", [])

if not available_models:
    raise RuntimeError(
        "The local Qwen server returned no available models."
    )

model_id = available_models[0]["id"]

print(f"Using model: {model_id}")

# Generate a response using conversation history.
assistant_text = generate_tutor_response(
    model_id=model_id,
    user_text=user_text,
)

llm_time = time.perf_counter() - start

print(f"\nLearner transcript: {user_text}")
print(f"Tutor response: {assistant_text}")
print(f"Qwen response time: {llm_time:.2f}s")
print(
    "Conversation messages stored: "
    f"{len(conversation_history)}/{MAX_HISTORY_MESSAGES}"
)


# ============================================================
# Stage 3: Text-to-speech
# ============================================================

print("\n[3/3] Initializing Kokoro...")

start = time.perf_counter()

tts_pipeline = KPipeline(lang_code="a")

tts_load_time = time.perf_counter() - start

print(f"Kokoro initialization: {tts_load_time:.2f}s")

print("Generating spoken response...")

start = time.perf_counter()

audio_chunks = []

with torch.inference_mode():
    for _, _, audio_chunk in tts_pipeline(
        assistant_text,
        voice=TTS_VOICE,
        speed=1.0,
        split_pattern=r"\n+",
    ):
        audio_chunks.append(
            audio_chunk.cpu().numpy()
        )

tts_time = time.perf_counter() - start

if not audio_chunks:
    raise RuntimeError(
        "Kokoro did not generate any audio."
    )

audio = np.concatenate(audio_chunks)

sf.write(
    str(AUDIO_OUTPUT),
    audio,
    SAMPLE_RATE,
)

audio_duration = len(audio) / SAMPLE_RATE

rtf = (
    tts_time / audio_duration
    if audio_duration
    else float("inf")
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("VOICE PIPELINE RESULTS")
print("=" * 60)

print(f"Recognized text: {user_text}")
print(f"Tutor reply: {assistant_text}")

print("\nConversation memory:")
print(f"Messages retained: {len(conversation_history)}")
print(f"Maximum messages: {MAX_HISTORY_MESSAGES}")
print("Memory type: In-memory, user/assistant messages")

print("\nTiming:")
print(f"Whisper initialization: {asr_load_time:.2f}s")
print(f"Whisper transcription:  {asr_time:.2f}s")
print(f"Qwen inference:         {llm_time:.2f}s")
print(f"Kokoro initialization:  {tts_load_time:.2f}s")
print(f"Kokoro generation:      {tts_time:.2f}s")
print(f"Generated audio length: {audio_duration:.2f}s")
print(f"TTS real-time factor:   {rtf:.3f}")

print(f"\nOutput audio: {AUDIO_OUTPUT}")

total_time = (
    asr_load_time
    + asr_time
    + llm_time
    + tts_load_time
    + tts_time
)

print(
    "\nFirst-response elapsed time, "
    f"including model initialization: {total_time:.2f}s"
)

print(
    "\nSubsequent-response estimate, excluding initialization: "
    f"{asr_time + llm_time + tts_time:.2f}s"
)

print("\nPipeline completed successfully.")
