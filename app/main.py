"""Loopback-only local web application for the English Speaking Tutor."""

from __future__ import annotations

import asyncio
import base64
import hmac
import io
import json
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.llm import LocalLLM
from app.memory import ConversationMemory
from app.retrieval.embeddings import LocalSentenceEmbedder
from app.store import SessionStore

ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger("english_tutor")
DATA = Path(os.environ.get("TUTOR_DATA_DIR", ROOT / "data"))
DATABASE = DATA / "tutor.sqlite3"
SYSTEM_PROMPT = (
    "You are a friendly, patient English-speaking tutor. Understand the learner's meaning. "
    "Gently correct meaningful mistakes, explain briefly, and keep spoken responses concise "
    "(2–4 sentences). Ask one natural follow-up when appropriate. Use plain spoken English."
)


class TurnRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    client_turn_id: str | None = Field(default=None, max_length=100)


class SpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


def clean_text_for_speech(text: str) -> str:
    text = re.sub(r"```.*?```", " Code example omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "link omitted", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"\*\*(.*?)\*\*|__(.*?)__", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)|(?<!_)_([^_]+)_(?!_)", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"(?m)^\s*[-+]\s+|^\s*\d+[.)]\s+", "", text)
    return re.sub(r"[*_~#]", "", text).strip()


def split_speech_sentences(text: str) -> list[str]:
    """Split normalized prose into complete, ordered sentences for Kokoro."""
    fragments = re.split(r"(?<=[.!?])[\"'”’)]*\s+", text.strip())
    return [fragment.strip() for fragment in fragments if fragment.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_dir = Path(getattr(app.state, "data_dir", DATA))
    data_dir.mkdir(parents=True, exist_ok=True)
    app.state.store = SessionStore(data_dir / "tutor.sqlite3")
    embedding_cache = Path(os.environ.get("TUTOR_EMBEDDING_CACHE", ROOT / "models" / "embeddings" / "cache"))
    app.state.embedder = LocalSentenceEmbedder(embedding_cache)
    try:
        if os.environ.get("TUTOR_NEURAL_MEMORY", "1").lower() not in {"0", "false", "no"}:
            await asyncio.to_thread(app.state.embedder.load)
            app.state.memory = ConversationMemory(data_dir / "conversation_memory.sqlite3", embedder=app.state.embedder)
            reindexed = await asyncio.to_thread(app.state.memory.reindex_neural)
            app.state.embedding_status = {"model": app.state.embedder.model_id, "device": "CPU", "mode": "neural", "reindexed": reindexed}
        else:
            raise RuntimeError("Neural memory disabled by TUTOR_NEURAL_MEMORY")
    except Exception as exc:
        LOGGER.warning("Neural embedding unavailable; using feature-hash fallback (%s: %s)", type(exc).__name__, exc)
        app.state.memory = ConversationMemory(data_dir / "conversation_memory.sqlite3")
        app.state.embedding_status = {"model": "feature-hash-v1", "device": "CPU", "mode": "lexical-fallback", "reason": f"{type(exc).__name__}: {exc}"}
    app.state.llm = LocalLLM()
    app.state.asr_model = None
    app.state.tts_pipeline = None
    app.state.model_lock = threading.Lock()
    app.state.memory_queue = asyncio.Queue(maxsize=256)
    app.state.memory_worker = asyncio.create_task(_memory_worker(app.state.memory_queue, app.state.memory))
    yield
    await app.state.memory_queue.join()
    app.state.memory_worker.cancel()
    try:
        await app.state.memory_worker
    except asyncio.CancelledError:
        pass


async def _memory_worker(queue: asyncio.Queue, memory: ConversationMemory):
    while True:
        user_text, assistant_text = await queue.get()
        try:
            await asyncio.to_thread(memory.add, user_text, assistant_text)
        except Exception:
            LOGGER.exception("Background memory indexing failed")
        finally:
            queue.task_done()


app = FastAPI(title="English Speaking Tutor", version="0.2.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "frontend"), name="static")


def services(request):
    return request.app.state.store, request.app.state.memory, request.app.state.llm


@app.get("/")
async def index():
    return FileResponse(ROOT / "frontend" / "index.html")


@app.get("/api/v1/health")
async def health():
    return {"ok": True, "app": "English Speaking Tutor", "version": app.version, "llama_server": os.environ.get("TUTOR_SERVER_URL", "http://127.0.0.1:8080"), "models": {"asr": "Whisper small (lazy, CPU)", "tts": "Kokoro af_heart (lazy, CPU)", "embedding": getattr(app.state, "embedding_status", {"mode": "starting"})}}


@app.post("/api/v1/admin/shutdown", include_in_schema=False)
async def request_shutdown(request: Request, token: str = Header(default="", alias="X-Tutor-Shutdown-Token")):
    expected = os.environ.get("TUTOR_SHUTDOWN_TOKEN", "")
    if not expected or not hmac.compare_digest(token, expected):
        raise HTTPException(403, "Invalid shutdown token.")
    server = getattr(request.app.state, "uvicorn_server", None)
    if server is None:
        raise HTTPException(503, "The managed server controller is unavailable.")
    server.should_exit = True
    return {"stopping": True}


@app.get("/api/v1/diagnostics")
async def diagnostics(request: Request):
    _, _, llm = services(request)
    try:
        model = await asyncio.to_thread(llm.model)
        status = {"available": True, "model": model}
    except Exception as exc:
        status = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {"llama_server": status, "embedding": getattr(request.app.state, "embedding_status", {"mode": "starting"}), "audio": "browser microphone permission and devices are checked when voice starts", "asr": {"model": "Whisper small", "device": "CPU", "loaded": request.app.state.asr_model is not None}, "tts": {"model": "Kokoro af_heart", "device": "CPU", "loaded": request.app.state.tts_pipeline is not None}}


@app.get("/api/v1/sessions")
async def list_sessions(request: Request):
    store, _, _ = services(request)
    return {"sessions": store.list_sessions()}


@app.post("/api/v1/sessions")
async def create_session(request: Request):
    store, _, _ = services(request)
    return store.create_session()


@app.get("/api/v1/sessions/{session_id}")
async def get_session(session_id: str, request: Request):
    store, _, _ = services(request)
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(404, "Session not found")
    return {**session, "messages": store.messages(session_id)}


@app.delete("/api/v1/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    store, _, _ = services(request)
    if not store.delete_session(session_id):
        raise HTTPException(404, "Session not found")
    return {"deleted": True}


@app.get("/api/v1/memories")
async def list_memories(request: Request, q: str = ""):
    _, memory, _ = services(request)
    items = await asyncio.to_thread(memory.search, q, 50) if q.strip() else await asyncio.to_thread(memory.recent, 50)
    return {"memories": items, "mode": request.app.state.embedding_status["mode"]}


@app.delete("/api/v1/memories/{memory_id}")
async def delete_memory(memory_id: int, request: Request):
    _, memory, _ = services(request)
    deleted = await asyncio.to_thread(memory.delete, memory_id)
    if not deleted:
        raise HTTPException(404, "Memory not found")
    return {"deleted": True}


async def process_turn(session_id: str, text: str, client_turn_id: str, store, memory, llm, memory_queue: asyncio.Queue) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise HTTPException(422, "Please enter or say something before sending.")
    prior = store.turn_messages(session_id, client_turn_id)
    prior_assistant = next((item for item in prior if item["role"] == "assistant"), None)
    prior_user = next((item for item in prior if item["role"] == "user"), None)
    if prior_assistant:
        answer = prior_assistant["content"]
        return {"user_text": prior_user["content"] if prior_user else text, "assistant_text": answer, "speech_text": clean_text_for_speech(answer), "client_turn_id": client_turn_id, "latency_seconds": 0, "replayed": True}
    store.add_message(session_id, "user", text, client_turn_id)
    # Context is bounded by approximate characters, not an arbitrary message count.
    history = store.messages(session_id)
    budget, prompt_messages = 12000, []
    used = len(SYSTEM_PROMPT)
    for message in reversed(history):
        cost = len(message["content"])
        if used + cost > budget:
            break
        prompt_messages.append({"role": message["role"], "content": message["content"]})
        used += cost
    prompt_messages.reverse()
    # An unusually long turn remains persisted but is clipped only in model context.
    if not prompt_messages or prompt_messages[-1]["role"] != "user":
        prompt_messages.append({"role": "user", "content": text[:budget]})
    memories = await asyncio.to_thread(memory.search, text, 3)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *prompt_messages]
    started = time.perf_counter()
    try:
        answer = await asyncio.to_thread(llm.reply, messages, memories)
    except Exception as exc:
        # Keep the learner's utterance even if the sidecar is unavailable.
        raise HTTPException(503, f"Your message was saved, but the local model is unavailable: {exc}") from exc
    store.add_message(session_id, "assistant", answer, client_turn_id)
    try:
        memory_queue.put_nowait((text, answer))
    except asyncio.QueueFull:
        LOGGER.error("Background memory queue is full; this completed turn remains in the session store but was not indexed")
    return {"user_text": text, "assistant_text": answer, "speech_text": clean_text_for_speech(answer), "client_turn_id": client_turn_id, "latency_seconds": round(time.perf_counter() - started, 3)}


@app.post("/api/v1/sessions/{session_id}/turns")
async def create_turn(session_id: str, body: TurnRequest, request: Request):
    store, memory, llm = services(request)
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    return await process_turn(session_id, body.text, body.client_turn_id or str(uuid4()), store, memory, llm, request.app.state.memory_queue)


@app.post("/api/v1/sessions/{session_id}/voice-turn")
async def create_voice_turn(session_id: str, request: Request):
    """Accept a short local WAV recording, transcribe locally, and answer."""
    store, memory, llm = services(request)
    if not store.get_session(session_id):
        raise HTTPException(404, "Session not found")
    raw = await request.body()
    if not raw or len(raw) > 20_000_000:
        raise HTTPException(413, "Audio is empty or exceeds the 20 MB limit.")
    try:
        import numpy as np
        import soundfile as sf
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000 or len(audio) < 4000:
            raise ValueError("Expected at least 0.25 seconds of 16 kHz mono WAV audio.")
    except Exception as exc:
        raise HTTPException(422, f"Could not read microphone audio: {exc}") from exc

    try:
        asr_model = await asyncio.to_thread(_get_asr_model, request.app)
        transcription = await asyncio.to_thread(asr_model.transcribe, audio, language="en", fp16=False, verbose=False)
        text = transcription["text"].strip()
    except Exception as exc:
        raise HTTPException(503, f"Local Whisper transcription failed: {type(exc).__name__}: {exc}") from exc
    if not text:
        raise HTTPException(422, "No speech was recognized. Try again closer to the microphone.")

    result = await process_turn(session_id, text, str(uuid4()), store, memory, llm, request.app.state.memory_queue)
    audio_base64 = None
    tts_error = None
    try:
        pipeline = await asyncio.to_thread(_get_tts_pipeline, request.app)
        audio_parts = await asyncio.to_thread(_synthesize, pipeline, result["speech_text"])
        if audio_parts:
            import numpy as np
            output = io.BytesIO()
            sf.write(output, np.concatenate(audio_parts), 24000, format="WAV", subtype="PCM_16")
            audio_base64 = base64.b64encode(output.getvalue()).decode("ascii")
    except Exception as exc:
        tts_error = f"Local speech synthesis failed: {type(exc).__name__}: {exc}"
    return {**result, "audio_base64": audio_base64, "tts_error": tts_error}


@app.post("/api/v1/voice/transcription")
async def transcribe_voice(request: Request):
    """Transcribe one WAV locally, returning verbatim text before model work."""
    raw = await request.body()
    if not raw or len(raw) > 20_000_000:
        raise HTTPException(413, "Audio is empty or exceeds the 20 MB limit.")
    try:
        import numpy as np
        import soundfile as sf
        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000 or len(audio) < 4000:
            raise ValueError("Expected at least 0.25 seconds of 16 kHz mono WAV audio.")
        asr_model = await asyncio.to_thread(_get_asr_model, request.app)
        transcription = await asyncio.to_thread(asr_model.transcribe, audio, language="en", fp16=False, verbose=False)
        text = transcription["text"].strip()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(503, f"Local Whisper transcription failed: {type(exc).__name__}: {exc}") from exc
    if not text:
        raise HTTPException(422, "No speech was recognized. Try again closer to the microphone.")
    return {"text": text}


@app.post("/api/v1/speech")
async def synthesize_text(body: SpeechRequest, request: Request):
    try:
        import numpy as np
        import soundfile as sf
        pipeline = await asyncio.to_thread(_get_tts_pipeline, request.app)
        audio_parts = await asyncio.to_thread(_synthesize, pipeline, clean_text_for_speech(body.text))
        if not audio_parts:
            raise RuntimeError("Kokoro returned no audio.")
        output = io.BytesIO()
        sf.write(output, np.concatenate(audio_parts), 24000, format="WAV", subtype="PCM_16")
        return {"audio_base64": base64.b64encode(output.getvalue()).decode("ascii")}
    except Exception as exc:
        raise HTTPException(503, f"Local speech synthesis failed: {type(exc).__name__}: {exc}") from exc


@app.post("/api/v1/speech/stream")
async def synthesize_text_stream(body: SpeechRequest, request: Request):
    """Stream sentence-sized WAV chunks as NDJSON with bounded synthesis work."""
    try:
        import numpy as np
        import soundfile as sf
        pipeline = await asyncio.to_thread(_get_tts_pipeline, request.app)
    except Exception as exc:
        raise HTTPException(503, f"Local speech synthesis failed: {type(exc).__name__}: {exc}") from exc
    sentences = split_speech_sentences(clean_text_for_speech(body.text))
    if not sentences:
        raise HTTPException(422, "There is no speakable text in this response.")

    async def chunks():
        try:
            for index, sentence in enumerate(sentences):
                parts = await asyncio.to_thread(_synthesize, pipeline, sentence)
                if not parts:
                    continue
                output = io.BytesIO()
                sf.write(output, np.concatenate(parts), 24000, format="WAV", subtype="PCM_16")
                yield (json.dumps({
                    "type": "audio",
                    "index": index,
                    "text": sentence,
                    "audio_base64": base64.b64encode(output.getvalue()).decode("ascii"),
                }) + "\n").encode("utf-8")
            yield (json.dumps({"type": "done", "chunk_count": len(sentences)}) + "\n").encode("utf-8")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            LOGGER.exception("Sentence-level local speech synthesis failed")
            yield (json.dumps({"type": "error", "message": f"Local speech synthesis failed: {type(exc).__name__}: {exc}"}) + "\n").encode("utf-8")

    return StreamingResponse(chunks(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})


def _synthesize(pipeline, text: str):
    import torch
    with torch.inference_mode():
        return [audio.detach().cpu().numpy() for _, _, audio in pipeline(text, voice="af_heart", speed=1.0) if audio is not None]


def _get_asr_model(application):
    with application.state.model_lock:
        if application.state.asr_model is None:
            import whisper
            application.state.asr_model = whisper.load_model("small", device="cpu", download_root=str(ROOT / "models" / "asr"))
        return application.state.asr_model


def _get_tts_pipeline(application):
    with application.state.model_lock:
        if application.state.tts_pipeline is None:
            from kokoro import KPipeline
            application.state.tts_pipeline = KPipeline(lang_code="a")
        return application.state.tts_pipeline


@app.websocket("/api/v1/ws/{session_id}")
async def session_socket(websocket: WebSocket, session_id: str):
    await websocket.accept()
    store = getattr(websocket.app.state, "store", None)
    if store is None or not store.get_session(session_id):
        await websocket.send_json({"type": "error", "message": "Session not found"})
        await websocket.close(code=4404)
        return
    memory, llm = websocket.app.state.memory, websocket.app.state.llm
    await websocket.send_json({"type": "state", "state": "idle"})
    turn_task = None
    receive_task = asyncio.create_task(websocket.receive_json())
    try:
        while True:
            waiting = {receive_task}
            if turn_task is not None:
                waiting.add(turn_task)
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
            if turn_task is not None and turn_task in done:
                try:
                    result = turn_task.result()
                    await websocket.send_json({"type": "assistant_text", **result})
                    await websocket.send_json({"type": "state", "state": "speaking"})
                except asyncio.CancelledError:
                    pass
                except HTTPException as exc:
                    await websocket.send_json({"type": "error", "message": str(exc.detail)})
                    await websocket.send_json({"type": "state", "state": "recovering"})
                turn_task = None
            if receive_task in done:
                event = receive_task.result()
                receive_task = asyncio.create_task(websocket.receive_json())
                if event.get("type") == "cancel":
                    if turn_task is not None:
                        turn_task.cancel()
                        turn_task = None
                    await websocket.send_json({"type": "state", "state": "interrupted"})
                    continue
                if event.get("type") != "user_text":
                    await websocket.send_json({"type": "error", "message": "Unsupported event"})
                    continue
                if turn_task is not None:
                    turn_task.cancel()
                turn_id = str(event.get("client_turn_id") or uuid4())
                user_text = str(event.get("text", ""))
                await websocket.send_json({"type": "transcript", "text": user_text, "final": True, "client_turn_id": turn_id})
                await websocket.send_json({"type": "state", "state": "thinking"})
                turn_task = asyncio.create_task(
                    process_turn(session_id, user_text, turn_id, store, memory, llm, websocket.app.state.memory_queue)
                )
    except WebSocketDisconnect:
        if turn_task is not None:
            turn_task.cancel()
        receive_task.cancel()
        return
