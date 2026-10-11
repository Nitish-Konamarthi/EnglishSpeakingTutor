import os
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import numpy as np
    import requests
    import sounddevice as sd
    import soundfile as sf
    import whisper
    from kokoro import KPipeline
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Required package '{exc.name}' is missing from Python "
        f"{sys.executable}. From the project root, run .\\run_tutor.ps1 "
        "or .\\.venv\\Scripts\\python.exe .\\scripts\\voice_tutor.py."
    ) from exc

from app.memory import ConversationMemory

if os.name == "nt":
    import msvcrt


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "asr"
MEMORY_DB = DATA_DIR / "conversation_memory.sqlite3"

RECORDING_PATH = DATA_DIR / "current_utterance.wav"
RESPONSE_PATH = DATA_DIR / "tutor_response.wav"

FFMPEG_BIN = (
    PROJECT_ROOT
    / "runtime"
    / "ffmpeg"
    / "ffmpeg-9.0.2-essentials_build"
    / "bin"
)

SERVER_URL = os.environ.get(
    "TUTOR_SERVER_URL",
    "http://127.0.0.1:8080",
).rstrip("/")

ASR_MODEL = "small"
TTS_VOICE = "af_heart"

SAMPLE_RATE = 16000
MAX_HISTORY_MESSAGES = 25
MAX_RETRIEVED_MEMORIES = 3
RETAIN_AUDIO = os.environ.get("TUTOR_RETAIN_AUDIO", "0").strip().lower() in {
    "1", "true", "yes", "on",
}

# Device indices from your sounddevice output.
MICROPHONE_DEVICE = 1
OUTPUT_DEVICE = 4

SYSTEM_PROMPT = """
You are a friendly, patient English speaking tutor.

Your goals:
- Help the learner speak English naturally and confidently.
- Continue conversations naturally using previous messages.
- Ask one clear follow-up question at a time.
- Correct important grammar mistakes gently.
- Explain corrections briefly using simple English.
- Encourage the learner without excessive praise.
- Do not turn every conversation into a grammar lesson.
- Keep spoken responses concise, ideally 2-4 sentences.
- Use natural, conversational English.
- Never claim to have heard something that was not transcribed.
- Do not use Markdown formatting; respond in natural spoken English.
- Use retrieved conversation memories only when they are relevant.
"""


# ============================================================
# ENVIRONMENT
# ============================================================

DATA_DIR.mkdir(parents=True, exist_ok=True)

os.environ["PATH"] = (
    str(FFMPEG_BIN)
    + os.pathsep
    + os.environ.get("PATH", "")
)


# ============================================================
# CONVERSATION HISTORY
# ============================================================

conversation_history = []
memory_store = ConversationMemory(MEMORY_DB)


def trim_history(history, max_messages=MAX_HISTORY_MESSAGES):
    """
    Keep the newest messages within the limit while preserving
    complete user/assistant turns.

    An unfinished user message is retained until the assistant
    response arrives.
    """
    if max_messages < 2:
        raise ValueError("max_messages must be at least 2.")

    # Retain only complete user/assistant pairs initially.
    complete_pairs = []

    for index in range(0, len(history) - 1, 2):
        user_message = history[index]
        assistant_message = history[index + 1]

        if (
            user_message["role"] == "user"
            and assistant_message["role"] == "assistant"
        ):
            complete_pairs.append(
                [user_message, assistant_message]
            )

    # Retain an unfinished user message at the end.
    pending_user = None

    if history and history[-1]["role"] == "user":
        pending_user = history[-1]

    # Keep the newest complete turns that fit within the limit.
    max_pairs = max_messages // 2
    retained_pairs = complete_pairs[-max_pairs:]

    result = [
        message
        for pair in retained_pairs
        for message in pair
    ]

    # If a user message is awaiting its assistant response,
    # reserve one slot for it.
    if pending_user is not None:
        result = result[-(max_messages - 1):]
        result.append(pending_user)

    return result

# ============================================================
# MICROPHONE RECORDING
# ============================================================

def record_utterance():
    """
    Press ENTER to start recording.
    Press ENTER again to stop recording.
    Type q at the first prompt to exit.
    """

    command = input(
        "\nPress ENTER to start speaking, or type q to quit: "
    ).strip().lower()

    if command == "q":
        return "QUIT"

    print("Recording... Press ENTER when you finish.")

    chunks = []

    def audio_callback(indata, frames, timing, status):
        if status:
            print(f"Audio status: {status}")

        chunks.append(indata.copy())

    try:
        with sd.InputStream(
            device=MICROPHONE_DEVICE,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            input()

    except KeyboardInterrupt:
        raise

    except Exception as exc:
        raise RuntimeError(
            f"Microphone recording failed: {exc}"
        ) from exc

    if not chunks:
        print("No audio was captured. Turn cancelled.")
        return None

    audio = np.concatenate(chunks, axis=0).reshape(-1)

    if len(audio) < SAMPLE_RATE // 2:
        print("Recording was too short. Turn cancelled.")
        return None

    sf.write(
        str(RECORDING_PATH),
        audio,
        SAMPLE_RATE,
        subtype="PCM_16",
    )

    duration = len(audio) / SAMPLE_RATE
    print(f"Recorded {duration:.1f} seconds.")

    return str(RECORDING_PATH)




# ============================================================
# SPEECH RECOGNITION
# ============================================================

def transcribe_audio(asr_model, audio_path):
    start = time.perf_counter()

    try:
        result = asr_model.transcribe(
            audio_path,
            language="en",
            fp16=False,
            verbose=False,
        )
    finally:
        if not RETAIN_AUDIO:
            try:
                Path(audio_path).unlink(missing_ok=True)
            except OSError as exc:
                print(f"Could not remove temporary recording: {exc}")

    transcript = result["text"].strip()

    elapsed = time.perf_counter() - start

    print(f"\nYou: {transcript}")
    print(f"Transcription time: {elapsed:.2f}s")

    return transcript


# ============================================================
# LOCAL LLM
# ============================================================

def get_model_id():
    response = requests.get(
        f"{SERVER_URL}/v1/models",
        timeout=10,
    )

    response.raise_for_status()

    models = response.json().get("data", [])

    if not models:
        raise RuntimeError(
            "No model is loaded in llama-server."
        )

    return models[0]["id"]


def generate_response(model_id, user_text):
    pending_history = [
        *conversation_history,
        {"role": "user", "content": user_text},
    ]
    history = trim_history(pending_history)

    memories = memory_store.search(user_text, limit=MAX_RETRIEVED_MEMORIES)
    memory_context = ""
    if memories:
        memory_context = "\n\nRelevant details from earlier sessions:\n" + "\n".join(
            f"- Learner: {item['user_text']} Tutor: {item['assistant_text']}"
            for item in memories
        )

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT + memory_context,
            },
            *history,
        ],
        "temperature": 0.4,
        "max_tokens": 180,
        "stream": False,
    }

    start = time.perf_counter()

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json=payload,
        timeout=180,
    )

    response.raise_for_status()

    data = response.json()

    tutor_text = (
        data["choices"][0]["message"]["content"].strip()
    )

    elapsed = time.perf_counter() - start

    if not tutor_text:
        raise RuntimeError("The tutor returned an empty response.")

    conversation_history[:] = history

    conversation_history.append(
        {
            "role": "assistant",
            "content": tutor_text,
        }
    )

    conversation_history[:] = trim_history(
        conversation_history
    )
    memory_store.add(user_text, tutor_text)

    print(f"\nTutor: {tutor_text}")
    print(f"LLM response time: {elapsed:.2f}s")
    print(
        f"Conversation messages: "
        f"{len(conversation_history)}/{MAX_HISTORY_MESSAGES}"
    )

    return tutor_text


def clean_text_for_speech(text):
    """Remove common Markdown syntax before sending text to Kokoro."""
    text = re.sub(r"```.*?```", " Code example omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*|__(.*?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?m)^\s*[-+]\s+|^\s*\d+[.)]\s+", "", text)
    text = re.sub(r"[*_~#]", "", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# TEXT-TO-SPEECH
# ============================================================

def synthesize_speech(tts_pipeline, text):
    start = time.perf_counter()

    audio_chunks = []

    for _, _, audio in tts_pipeline(
        text,
        voice=TTS_VOICE,
        speed=1.0,
    ):
        if audio is not None:
            audio_chunks.append(
                audio.detach().cpu().numpy()
            )

    if not audio_chunks:
        raise RuntimeError(
            "Kokoro did not generate any audio."
        )

    audio_output = np.concatenate(audio_chunks)

    if RETAIN_AUDIO:
        sf.write(
            str(RESPONSE_PATH),
            audio_output,
            24000,
            subtype="PCM_16",
        )

    elapsed = time.perf_counter() - start

    duration = len(audio_output) / 24000

    print(f"TTS generation time: {elapsed:.2f}s")
    print(f"Generated audio duration: {duration:.2f}s")

    return audio_output


def play_response(audio):
    """
    Play tutor audio.

    Press ENTER to interrupt playback.
    Pressing Enter does not end the conversation.
    """
    print("\nPlaying tutor response...")
    print("Press ENTER to interrupt playback.")

    interrupted = False
    try:
        sd.play(
            audio,
            samplerate=24000,
            device=OUTPUT_DEVICE,
        )

        playback_started = time.monotonic()
        playback_duration = len(audio) / 24000
        while time.monotonic() - playback_started < playback_duration:
            if os.name == "nt" and msvcrt.kbhit():
                if msvcrt.getwch() == "\r":
                    interrupted = True
                    break
            time.sleep(0.05)

        if not interrupted:
            sd.wait()

    finally:
        sd.stop()

    if interrupted:
        print("Playback interrupted.")
    else:
        print("Playback finished.")



# ============================================================
# MAIN CONVERSATION LOOP
# ============================================================

def main():
    print("=" * 60)
    print("ENGLISH SPEAKING TUTOR")
    print("MULTI-TURN VOICE CONVERSATION")
    print("=" * 60)

    print("\nLoading Whisper small...")

    start = time.perf_counter()

    asr_model = whisper.load_model(
        ASR_MODEL,
        device="cpu",
        download_root=str(MODEL_DIR),
    )

    print(
        f"Whisper loaded in "
        f"{time.perf_counter() - start:.2f}s"
    )

    print("\nLoading Kokoro...")

    start = time.perf_counter()

    tts_pipeline = KPipeline(lang_code="a")

    print(
        f"Kokoro initialized in "
        f"{time.perf_counter() - start:.2f}s"
    )

    print("\nChecking local Qwen server...")

    model_id = get_model_id()

    print(f"Connected to model: {model_id}")

    print("\nTutor ready.")
    print("Speak naturally in English.")
    print("Press Ctrl+C to exit.")
    print("")

    try:
        while True:
            try:
                audio_path = record_utterance()

                if audio_path == "QUIT":
                    print("Ending conversation. Goodbye!")
                    break

                if audio_path is None:
                    continue

                user_text = transcribe_audio(
                    asr_model,
                    audio_path,
                )

                if not user_text:
                    print(
                        "I couldn't recognize your speech. "
                        "Please try again."
                    )
                    continue

                if user_text.strip().lower() in {
                    "quit", "exit", "stop the session", "goodbye",
                }:
                    print("Ending conversation. Goodbye!")
                    break

                tutor_text = generate_response(
                    model_id,
                    user_text,
                )

                audio = synthesize_speech(
                    tts_pipeline,
                    clean_text_for_speech(tutor_text),
                )

                play_response(audio)

                print("\n" + "-" * 60)

            except (RuntimeError, ValueError) as exc:
                print(f"\nCould not complete this turn: {exc}")
                print("You can try speaking again.")

    except KeyboardInterrupt:
        print("\n\nConversation ended.")

    finally:
        sd.stop()

        print(
            f"Final conversation history: "
            f"{len(conversation_history)} messages"
        )

        print("Tutor stopped.")


if __name__ == "__main__":
    main()
