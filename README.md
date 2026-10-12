# English Speaking Tutor

A local-first English practice app for Windows. It includes a browser interface, persistent sessions, text chat, hands-free microphone capture with local Whisper transcription, Kokoro speech output, and local semantic conversation retrieval. The recommended launcher manages the bundled local llama.cpp server when one is not already serving the configured Qwen model.

## Run it

Requirements: Windows, Python 3.12, the project `.venv`, and the checked-in Qwen GGUF and llama.cpp runtime. Whisper small and Kokoro weights are loaded locally on first use; neural retrieval weights are optional and use the offline cache when present.

From the project root, install the project dependencies once if needed, then use the single recommended command:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\run_app.ps1
```

The launcher first checks `http://127.0.0.1:8080/v1/models`. It reuses a healthy server only when its model identifier matches the configured Qwen GGUF; otherwise, it starts `runtime\llama.cpp\llama-server.exe` with `models\llm\Qwen3-4B-Instruct-2507-Q4_K_M.gguf`, waits for API readiness, then starts and checks the web app. It binds both services to loopback. Logs are written to `logs\`. Press Ctrl+C in the launcher window to stop the web backend and any llama-server process started by that launcher. An already-running external server is never stopped.

Open **http://127.0.0.1:8765**. Optional configuration (set before launch): `$env:TUTOR_APP_PORT = "8766"`, `$env:TUTOR_SERVER_URL = "http://127.0.0.1:8081"`, `$env:TUTOR_LLM_MODEL = "D:\models\qwen.gguf"`, `$env:TUTOR_LLAMA_SERVER_EXE = "D:\llama.cpp\llama-server.exe"`, `$env:TUTOR_CONTEXT_SIZE = "8192"`, `$env:TUTOR_GPU_LAYERS = "20"`, `$env:TUTOR_SERVER_TIMEOUT_SECONDS = "300"`, `$env:TUTOR_MANAGE_LLAMA = "0"` to require an already-running server, and `$env:TUTOR_OPEN_BROWSER = "0"` to suppress browser launch. Additional llama.cpp arguments can be provided as a JSON string array in `TUTOR_LLAMA_ARGS_JSON`. The default server URL must remain loopback; non-local addresses are rejected.

The existing terminal prototype remains available as `.\run_tutor.ps1`.

## What works in this build

- Browser text conversation, a responsive local UI, and a review/search/delete panel for persisted conversation memories.
- Persistent sessions and complete user/assistant messages in `data/tutor.sqlite3`.
- Text and microphone input. The browser calibrates the room noise for 750 ms on startup, then applies an RMS speech detector with 250 ms steady-state pre-roll, a 250 ms sustained onset check, and three seconds of continuous silence before finalizing. A longer initial buffer preserves speech that begins during calibration. The default adaptive noise margin is 0.03 (at least 0.045 RMS, or about 4.6× over the calibrated floor); the versioned setting resets prior lower browser preferences. The live status reports mic level, calibrated floor, speech trigger, silence countdown, and capture errors. The user-adjustable noise margin trades quiet-speech detection against room-noise rejection. Completed turns are sent as 16 kHz WAV to Whisper small on CPU. The detector retains at most 90 seconds per segment and then starts a fresh segment if speech continues.
- Manual stop finalizes and submits buffered speech rather than clearing it. Empty and very short utterances are ignored. The microphone stays active between turns; detected speech stops current playback promptly and captures the new utterance. Browser echo cancellation is requested, but speaker playback may still leak into the microphone; headphones improve isolation.
- Kokoro `af_heart` CPU speech for voice turns, with ordered sentence-sized WAV chunks streamed to the browser as each is synthesized. Playback can start before later sentences finish. LLM text generation still completes before TTS begins; there is no token streaming.
- Local Qwen chat through the existing OpenAI-compatible sidecar. The learner message is saved before inference, so a sidecar failure does not lose it.
- Neural `sentence-transformers/all-MiniLM-L6-v2` embeddings on CPU, SQLite FTS5 plus `sqlite-vec` cosine search, and rank fusion. Existing feature-hash memories are re-embedded at app startup when the local model cache is present.
- A bounded background queue for durable episode-memory indexing.
- Loopback health and diagnostics endpoints: `/api/v1/health` and `/api/v1/diagnostics`.
- Local sessions, model weights, transcripts and memory. No audio or transcript content is sent to cloud services by this app.

## Model and placement choices

Qwen runs through the local llama.cpp server. Whisper and Kokoro run on CPU. MiniLM embeddings also run on CPU; the app does not load another model onto the 4 GB RTX 3050. This keeps retrieval local while limiting competition for GPU memory. The app reports embedding fallback status if cached weights or the vector extension are unavailable.

The embedding weights are cached under `models/embeddings/cache` and are ignored by Git. Download them once with network access if they are not already present:

```powershell
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu', cache_folder='models/embeddings/cache')"
```

The checked-in retrieval fixture currently has five labelled paraphrase queries. Run `.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py` for the measured recall@3 and query latency JSON in `reports/retrieval_eval.json`. This is a small smoke evaluation, not a production-quality benchmark. A cross-encoder reranker is deliberately disabled until a larger labelled set shows a quality benefit worth its added CPU latency.

## Data, privacy and recovery

- Sessions/messages: `data/tutor.sqlite3`.
- Conversation retrieval index: `data/conversation_memory.sqlite3`.
- ASR and embedding weights: `models/`.
- Audio: microphone audio is processed in memory and is not retained by the web app. The terminal prototype supports optional WAV retention with `TUTOR_RETAIN_AUDIO=1`.
- Database backups: stop the app and copy the two SQLite files. To clear history and memory, stop the app and remove those two files.
- If neural weights are missing, the app starts with the feature-hash/FTS fallback and reports that mode in diagnostics. For fully offline operation, download weights before disconnecting from the network.

## Voice controls

Press the microphone once to start hands-free conversation. Speak naturally; after the silence interval, the app transcribes and sends the utterance and then returns to listening. The status shows actual input energy so a quiet or stalled microphone is distinguishable from speech detection. Press the active microphone button to stop listening; any current captured utterance is finalized and submitted first. Voice settings allow selecting an input device, output device where the browser supports `setSinkId`, spoken replies, noise margin, and end-of-turn delay. Lower the noise margin to catch quieter speech relative to the measured room noise. The selected settings are saved in that browser. To use text only, do not grant microphone permission and use the message composer. Escape stops current playback or cancels a WebSocket response.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
node scripts\test_frontend_voice.js
.\.venv\Scripts\python.exe -m py_compile app\main.py app\llm.py app\memory.py app\store.py app\retrieval\embeddings.py
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py
```

`scripts/test_frontend_voice.js` exercises silence timing, pause reset, manual stop flush, short/empty utterance rejection, and bounded long speech. `scripts/test_voice_pipeline.py` remains the original ASR → Qwen → Kokoro regression reference. It overwrites `data/pipeline_response.wav`; preserve that file before running. The pipeline now uses the configured local server and requires the app's Qwen server to be ready. Browser microphone behavior still needs a real browser and audio device; deterministic tests do not prove hardware behavior.

## Implemented versus outstanding

See [docs/architecture-status.md](docs/architecture-status.md) for phase results and evidence. This build does not yet provide token streaming, cancellation of in-flight llama.cpp generation, a neural VAD/Smart Turn, background fact extraction/consolidation/retries, editable fact memory, vocabulary scheduling, LanguageTool feedback, interview scorecards, or a 20-minute microphone/GPU stress report. The browser audio pipeline still needs real microphone/barge-in verification; those features are not claimed complete.
