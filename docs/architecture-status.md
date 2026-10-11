# Architecture review and implementation status

Reviewed against the supplied **English Speaking Tutor — Final Architecture & Tech Stack** (including its native-Windows revision), the attached Claude artifact, and the referenced ChatGPT branch.

## Product being built

A private, single-user English-speaking tutor and mock-interview coach that runs on the Windows laptop. It should preserve verbatim speech evidence, give concise spoken feedback, remember learner facts and recurring language issues across sessions, and later provide drills, interview scorecards, fluency trends, and an editable progress dashboard. Voice, conversation data, and inference stay local after model downloads.

The intended deployment is a small FastAPI/async application plus the local llama.cpp HTTP sidecar. The current code is still a push-to-talk CLI prototype, not that final product.

## Branch status against the target

| Area | Current branch | Target / next gate |
|---|---|---|
| Runtime | Python CLI using the existing `.venv` | Python 3.12 modular app; later FastAPI + WebSocket |
| LLM | Qwen3-4B GGUF through the already-running llama-server at port 8080 | Keep model behind config; compare Qwen3.5 only in a measured bake-off |
| Speech input | Push-to-talk recording, Whisper small on CPU | Silero VAD + Smart Turn + verbatim ASR after a native-Windows feasibility/quality spike |
| Speech output | Kokoro synthesizes the full response before playback | Sentence-level streaming and barge-in |
| Session context | Bounded in-memory turn history | Working memory plus persistent episodic/fact/learner layers |
| Retrieval | SQLite + FTS5 and feature-hash cosine vectors | EmbeddingGemma by default (MiniLM lite), SQLite FTS5 + sqlite-vec, RRF and retrieval evaluation |
| Reranking | Not on the live path | Optional CPU cross-encoder only if Phase 2 evaluation shows false matches; do not spend an extra Qwen call on every spoken turn |
| Learning features | Gentle corrections from the prompt | LanguageTool-assisted error history, vocabulary/SRS, deterministic fluency metrics |
| Interview practice | Not implemented | State machine, rubric coverage, follow-ups, scorecards |
| UI | Terminal | Local browser app, transcripts, dashboard, inspectable/editable memory |
| Evaluation | Focused local memory unit tests | Latency/VRAM measurements, ASR WER/filler recall, retrieval recall@3, interview rubric agreement |

## Hardware and placement

The supplied machine is an i5-12450H, 16 GB DDR4-3200, RTX 3050 Laptop GPU with 4 GB VRAM and 60 W TGP. A read-only `nvidia-smi` sample while llama-server was already running showed 3,115 MiB used and about 851 MiB free. This is one live snapshot, not an idle baseline or a long-session peak.

That makes the current CPU placement for Whisper and Kokoro a sensible safe default. Do not load GPU ASR or GPU TTS beside the current server until a measured 20-minute peak stays below the 3.8 GB target. Keep the Python pipeline CUDA-free where possible; the LLM and its KV cache are the highest-value GPU resident components.

Prefill and decode have different bottlenecks. Prompt prefill processes many tokens in parallel and is primarily compute-bound. Decode generates one token at a time and repeatedly reads model weights, so it is primarily memory-bandwidth-bound. The report's token-rate and latency numbers are estimates, not measurements from this laptop. For this use case, prompt size, end-of-turn detection, ASR, and time to first TTS audio matter more than maximum decode throughput.

## Decisions and trade-offs

- **Keep the current sidecar user-managed.** It is already running and the user explicitly asked not to start another one. The CLI checks the OpenAI-compatible API and connects to it. A future managed-child option should be opt-in and must detect an existing server first.
- **Keep Qwen3-4B Q4_K_M as the current baseline.** It is already loaded and about 2.5 GB of weights fits better than larger voice models. Qwen3.5 is a candidate for a controlled quality/VRAM bake-off, not an automatic replacement.
- **Keep ASR on CPU for now.** Whisper small is the known-good current implementation. CrisperWhisper's verbatim output and timestamps better fit a speaking coach, but the report itself flags Windows backend support and 4 GB VRAM as risks. Compare on the learner's own recordings before switching.
- **Keep Kokoro on CPU and remove audio retention by default.** This preserves VRAM and privacy. WAV retention is now opt-in for debugging.
- **Do not call the chat model as a reranker during every live turn.** That adds a second LLM request and delays speech. First evaluate hybrid retrieval; add the small CPU cross-encoder only if it improves labelled retrieval cases.
- **Treat current vectors as a prototype.** Feature hashing is deterministic and offline but lexical; it is not a semantic embedding model. The README calls this out. Replace it with the selected local embedding model and `sqlite-vec` as a dedicated Phase 2 change.

## Phased build gates

1. **Phase 0 — stabilize and measure:** current push-to-talk pipeline, robust environment setup, model endpoint check, latency/VRAM records. This branch still needs repeatable end-to-end measurements.
2. **Phase 1 — voice experience:** VAD/turn detection, live transcripts, sentence-streamed Kokoro, reliable cancellation/barge-in, session persistence, ASR and LLM comparisons.
3. **Phase 2 — learner memory:** neural embeddings, sqlite-vec + FTS5/RRF, facts and error/vocabulary models, consolidation provenance, SRS, memory review/edit/delete, labelled retrieval evaluation.
4. **Phase 3 — mock interviews:** topic/role setup, question state machine, rubric coverage and correctness checks, follow-ups and scorecards.
5. **Phase 4 — dashboard and grammar service:** trends, LanguageTool, and optional pronunciation hints.

The final acceptance targets in the supplied report remain: p50 latency ≤2.5 s, p95 ≤4 s, peak VRAM ≤3.8 GB, ASR filler recall ≥80%, retrieval recall@3 ≥0.8, and interview score agreement ≥80%. None should be presented as achieved until measured against the specified fixtures and user-labelled examples.

## Bugs addressed in this pass

- The reported `ModuleNotFoundError: whisper` came from system Python (`C:\Users\91939\AppData\Local\Programs\Python\Python312\python.exe`), while Whisper is installed in the project `.venv`. Added `run_tutor.ps1` to select that environment and made direct wrong-interpreter launches print the right command.
- Fixed SQLite connection lifecycle: the connection context now commits/rolls back and closes the database handle, avoiding locked database files on Windows.
- Added FTS5/vector candidate fusion and rejected weak random hash matches.
- Removed live Qwen reranking from the response path to honor the latency and GPU-sharing constraints.
- Fixed the playback interrupt's background `input()` race; Windows key polling no longer competes with the next recording prompt.
- Audio files are deleted after transcription and generated audio is not written unless `TUTOR_RETAIN_AUDIO=1`.
- Corrected the `q` exit path and removed the duplicate spoken-quit check.

The separate Claude export archive linked from the manifest was not available to inspect: Claude returned a Cloudflare browser challenge to the approved download request. The supplied architecture report and Claude summary artifact were readable and used above; no other manifest archives were fetched.
