"""Exercise the app ASR → local Qwen → Kokoro path with the saved fixture.

Requires the user's already-running llama-server. Does not start or stop it and
does not write microphone or synthesized audio to disk.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import soundfile as sf
import numpy as np
import whisper
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FFMPEG_BIN = ROOT / "runtime" / "ffmpeg" / "ffmpeg-9.0.2-essentials_build" / "bin"
if FFMPEG_BIN.exists():
    os.environ["PATH"] = str(FFMPEG_BIN) + os.pathsep + os.environ.get("PATH", "")

from app.main import app


def main() -> int:
    # The tracked fixture is ADTS audio stored with a .wav suffix; use
    # Whisper/ffmpeg to decode and package a real mono 16 kHz WAV for the API.
    fixture_audio = whisper.load_audio(str(ROOT / "data" / "asr_test.wav"))
    fixture_buffer = io.BytesIO()
    sf.write(fixture_buffer, np.asarray(fixture_audio), 16000, format="WAV", subtype="PCM_16")
    fixture = fixture_buffer.getvalue()
    (ROOT / "tmp").mkdir(exist_ok=True)
    report_path = ROOT / "reports" / "app_voice_smoke.json"
    report_path.parent.mkdir(exist_ok=True)
    start = time.perf_counter()
    report = {"fixture": "data/asr_test.wav", "llama_endpoint": "local configured endpoint", "audio_saved": False}
    with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as directory:
        app.state.data_dir = Path(directory)
        try:
            with TestClient(app) as client:
                startup_seconds = time.perf_counter() - start
                session = client.post("/api/v1/sessions").json()
                stage = time.perf_counter()
                transcript = client.post("/api/v1/voice/transcription", content=fixture, headers={"Content-Type": "audio/wav"})
                if transcript.is_error:
                    raise RuntimeError(f"Whisper endpoint failed ({transcript.status_code}): {transcript.text}")
                asr_seconds = time.perf_counter() - stage
                stage = time.perf_counter()
                turn = client.post(
                    f"/api/v1/sessions/{session['id']}/turns",
                    json={"text": transcript.json()["text"], "client_turn_id": "fixture-smoke-turn"},
                )
                turn.raise_for_status()
                llm_seconds = time.perf_counter() - stage
                stage = time.perf_counter()
                speech = client.post("/api/v1/speech/stream", json={"text": turn.json()["speech_text"]})
                speech.raise_for_status()
                events = [json.loads(line) for line in speech.text.splitlines()]
                if events[-1].get("type") != "done":
                    raise RuntimeError(f"Sentence speech stream did not complete: {events[-1] if events else 'no chunks'}")
                wavs = [base64.b64decode(item["audio_base64"]) for item in events if item.get("type") == "audio"]
                if not wavs:
                    raise RuntimeError("Sentence speech stream returned no audio chunks.")
                tts_seconds = time.perf_counter() - stage
                chunks = [sf.read(io.BytesIO(wav), dtype="float32") for wav in wavs]
                sample_rate = chunks[0][1]
                if any(rate != sample_rate for _, rate in chunks):
                    raise RuntimeError("Kokoro sentence chunks used inconsistent sample rates.")
                audio = np.concatenate([part for part, _ in chunks])
                report.update({
                    "result": "pass",
                    "model": client.get("/api/v1/diagnostics").json()["llama_server"].get("model"),
                    "embedding_mode": app.state.embedding_status["mode"],
                    "startup_seconds": round(startup_seconds, 3),
                    "asr_seconds_including_first_load": round(asr_seconds, 3),
                    "qwen_seconds": round(llm_seconds, 3),
                    "kokoro_seconds_including_first_load": round(tts_seconds, 3),
                    "complete_response_seconds_after_upload": round(asr_seconds + llm_seconds + tts_seconds, 3),
                    "wav_sample_rate": sample_rate,
                    "generated_audio_seconds": round(len(audio) / sample_rate, 3),
                    "sentence_audio_chunks": len(wavs),
                    "response_characters": len(turn.json()["assistant_text"]),
                })
        finally:
            if hasattr(app.state, "data_dir"):
                del app.state.data_dir
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
