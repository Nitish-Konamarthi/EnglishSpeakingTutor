# Architecture and implementation status

Last updated: 2026-10-12. Implemented behavior and outstanding targets are distinguished below.

## Product and hardware

This is a private, local English-speaking tutor for the Windows laptop. The supplied architecture and Claude summary selected Qwen3-4B through llama.cpp, Whisper small and Kokoro on CPU, a browser interface, durable sessions, layered learner memory, neural retrieval, and optional—not automatic—reranking.

The project environment is Python 3.12.10. The hardware report and live snapshot identify an i5-12450H-class CPU, 16 GB RAM, and an RTX 3050 Laptop GPU with 4096 MiB VRAM / 60 W TGP. With the bundled llama-server running, an `nvidia-smi` sample reported 2189 MiB used and 0% utilization. This is a snapshot, not a peak. Whisper, Kokoro and MiniLM run on CPU.

## Current data flow

```text
Browser text ──────────────────────────────┐
Browser mic → RMS VAD + 3 s silence endpoint → WAV → Whisper small (CPU) ─┤
                                                          ↓
                         FastAPI → durable session SQLite → bounded prompt + retrieval
                                                          ↓
                                    existing llama-server /v1/chat/completions
                                                          ↓
                  persisted assistant turn → Kokoro sentence WAV stream → bounded browser queue
                                                          ↓
                              bounded background queue → neural episode index
```

The app and managed llama-server bind to loopback. `run_app.ps1` reuses a compatible Qwen server or starts only its own configured server process, checks model readiness, starts the app, and cleans up owned processes on shutdown. Session messages are stored in `data/tutor.sqlite3`. Episode retrieval uses `data/conversation_memory.sqlite3`: FTS5 lexical candidates + `sqlite-vec` cosine distance over MiniLM embeddings + reciprocal-rank fusion. Existing feature-hash memories are version-tagged and re-embedded on startup. Prompt history uses a character budget rather than the terminal prototype’s fixed 25-message limit.

## Phase status

| Phase | Status | Evidence and remaining work |
|---|---|---|
| A — Audit and stabilize | Partial | Existing changes preserved. The original voice-pipeline reference was run and its pre-existing WAV restored. Empty audio is rejected, markdown/URLs are normalized for speech, learner messages persist before model inference, and database handles close correctly. The new PowerShell launcher was exercised through managed start, compatible-server reuse, and managed shutdown. |
| B — API and sessions | Implemented for current scope, API tested | FastAPI lifecycle, loopback diagnostics, REST session CRUD/text turns, WebSocket state/transcript/reply/cancel events, SQLite session store, duplicate-turn protection, memory review/delete endpoints, token-protected graceful shutdown, and clear sidecar failure handling. Latest full automated suite: 17/17 passed. |
| C — Voice interaction | Partial | Browser RMS detection calibrates the room-noise floor for 750 ms, reports live microphone level/speech trigger/silence countdown, and watches for stalled/disconnected capture. It uses 250 ms steady-state pre-roll, a 250 ms sustained onset check, a three-second continuous-silence default, and bounded 90-second segments. The default noise margin is now 0.03 (minimum 0.045 RMS, about 4.6× over the calibrated floor); a v5 setting key resets prior lower preferences. Deterministic regressions cover steady 0.006, 0.009 and 0.052 RMS room noise, brief transients, 0.026 spikes and sustained 0.04 ambient noise over a 0.003 floor, as well as speech and the silence endpoint. The browser confirmed mic startup after correcting the missing voice-settings markup. Room input varied from 0.005 to 0.009 RMS; a separate check saw a 0.026 RMS spike, prompting a higher default. The full local-ASR → Qwen → Kokoro saved-fixture smoke passed. A deliberate spoken phrase and physical barge-in remain unverified. No Silero/Smart Turn or partial ASR. |
| D — Streaming TTS | Implemented for sentence chunks | Kokoro `af_heart` runs one complete sentence at a time and the API streams ordered WAV chunks. Browser playback starts per chunk, queues at most two waiting chunks, and aborts/clears old audio on detected speech or Escape. LLM text still completes before TTS begins; no token streaming or in-flight llama.cpp cancellation. |
| E — Neural retrieval | Implemented for current scope | `sentence-transformers/all-MiniLM-L6-v2` CPU embeddings, versioned rows, backfill, `sqlite-vec` cosine table, FTS5, RRF and a 0.25 neural similarity threshold. Fixture recall@3 is 5/5 (1.0) over five queries; the small fixture is only a smoke check. |
| F — Background jobs/reranker | Partial | A bounded 256-item queue indexes completed turns, logs worker failures without transcript text and drains on shutdown. No extraction/consolidation/retry/dead-letter workflow yet. Cross-encoder is disabled because five labels do not justify added CPU cost; no impact claim is made. |
| G — Frontend | Partial | Responsive UI, session history, text, hands-free mic, state display, local playback interruption, model diagnostics, memory review/search/delete, and browser-persisted voice settings for silence, sensitivity, devices, and spoken replies. The sidebar button now collapses the desktop sidebar and opens/closes it on narrow screens. Five consecutive real local-model text turns, memory review, settings, and restore after app restart were browser-verified. Narrow viewport and accessibility automation are still outstanding. |
| H — Learning | Not implemented | No evidence-backed grammar patterns, vocabulary scheduler, LanguageTool, deterministic fluency trend or interview scorecards. |
| I — Soak/handoff | Not complete | Original and updated fixture pipelines passed. The actual microphone was opened but no known phrase was spoken; barge-in and playback recovery were not hardware-verified. No 20-minute soak, peak RAM/VRAM or latency percentile test. Do not treat this build as production-ready. |

## Measured results

- Baseline automated suite: 14/14. After changes: 17/17 Python tests, 9 deterministic JavaScript voice lifecycle checks, JS syntax checks, and Python compilation passed. The sandbox blocks Windows asyncio socket-pair creation, so FastAPI tests were run outside it.
- Whisper small CPU: updated app smoke took 4.725 s on the saved fixture including first load. The original pipeline regression took 5.61 s initialization + 8.11 s transcription. These are individual runs, not percentiles.
- Qwen local model: updated app smoke took 3.031 s; the original pipeline regression took 5.12 s. Five browser text turns all returned useful answers. Token streaming/TTFT is not measured.
- Kokoro CPU original full pipeline: 24.63 s generation for 27.45 s audio (RTF 0.897), after 6.35 s initialization. Updated sentence-stream smoke: 10.647 s total for 19.75 s of audio in six chunks (RTF 0.539), in a separate run. This is not a controlled same-text A/B comparison.
- Real streaming API timing for two sentences: cold first WAV chunk 3.937 s and complete synthesis 4.895 s; warm repeat first chunk 0.531 s and complete synthesis 1.496 s. These are audio-ready timings, not audible-playback latency. Combined fixture ASR + Qwen + all TTS chunks took 18.403 s; first-audio end-to-end time was not measured.
- Neural retrieval: 5/5 labelled paraphrases in top three; 31.51 ms mean query latency in this run on Windows 11 / Python 3.12.10 CPU. The five-query set is only a smoke evaluation.
- Current GPU sample: 2189 MiB / 4096 MiB used and 0% utilization. `Get-Process` samples showed ~3386 MiB llama-server and ~2535 MiB app Python working sets. These are point-in-time working sets, not peak RAM/VRAM.
- No reranker comparison was run.

## Decisions

- Use the existing Qwen GGUF and bundled llama.cpp executable. Default URL is `http://127.0.0.1:8080`; override model, binary and endpoint through the launcher's documented environment variables.
- CPU placement for ASR/TTS/embeddings is safer than adding GPU workloads beside llama.cpp on a 4 GB card.
- MiniLM is the implemented compact neural baseline. EmbeddingGemma remains a candidate for a larger labelled bake-off.
- App defaults to `127.0.0.1:8765`, stores data locally, retains no audio, and avoids transcript content in diagnostics/logs.
- If cached neural weights are absent, startup reports the feature-hash fallback and its reason.

## Run and verify

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\run_app.ps1
```

Open `http://127.0.0.1:8765`. First launch loads cached MiniLM weights. Use `.\.venv\Scripts\python.exe -m pytest -q`, `node scripts\test_frontend_voice.js`, `scripts\evaluate_retrieval.py`, and `scripts\smoke_app_pipeline.py` for local checks. The smoke pipeline requires the configured server and saves no audio; `scripts\test_voice_pipeline.py` overwrites `data\pipeline_response.wav`, so back it up first.

## Outstanding release gates

Still outstanding: cancel in-flight llama.cpp generation; compare VAD against background noise and verify intentional barge-in/repeated interruption; verify audio device choices on target browsers; learner-fact extraction/consolidation and editable/exportable facts; retryable background maintenance; larger retrieval/reranker evaluation; learning progress/interview flows; automated browser checks across responsive layouts; latency percentiles; peak RAM/VRAM; and a long-session soak. Keep `scripts/test_voice_pipeline.py` as the regression reference and preserve its generated WAV output when running it.
