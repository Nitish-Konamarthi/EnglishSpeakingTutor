import tempfile
import time
import json
from types import SimpleNamespace
from pathlib import Path
import pytest

from fastapi.testclient import TestClient

from app import main


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def model(self):
        return "fake-local-model"

    def reply(self, messages, memories=None):
        self.calls += 1
        assert messages[-1]["role"] == "user"
        return "That sounds interesting. What do you enjoy most about it?"


def test_text_turn_persists_and_returns_spoken_text(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.setenv("TUTOR_NEURAL_MEMORY", "0")
        memory = None
        session_id = None
        with TestClient(main.app) as client:
            fake_llm = FakeLLM()
            client.app.state.llm = fake_llm
            memory = client.app.state.memory
            assert "English Speaking Tutor" in client.get("/").text
            assert "getUserMedia" in client.get("/static/app.js").text
            session = client.post("/api/v1/sessions").json()
            session_id = session["id"]
            body = {"text": "I enjoy to cycle.", "client_turn_id": "turn-1"}
            response = client.post(f"/api/v1/sessions/{session['id']}/turns", json=body)
            replay = client.post(f"/api/v1/sessions/{session['id']}/turns", json=body)
            assert response.status_code == 200
            assert replay.json()["replayed"] is True
            assert fake_llm.calls == 1
            saved = client.get(f"/api/v1/sessions/{session['id']}").json()["messages"]
            assert [item["role"] for item in saved] == ["user", "assistant"]
        assert memory is not None and session_id is not None
        indexed = memory.recent(1)[0]
        assert indexed["source_session_id"] == session_id


def test_user_message_is_saved_when_local_llm_is_offline(monkeypatch):
    class BrokenLLM:
        def reply(self, *_args, **_kwargs):
            raise ConnectionError("connection refused")

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.setenv("TUTOR_NEURAL_MEMORY", "0")
        with TestClient(main.app) as client:
            client.app.state.llm = BrokenLLM()
            session_response = client.post("/api/v1/sessions")
            assert session_response.status_code == 200, session_response.text
            session = session_response.json()
            response = client.post(
                f"/api/v1/sessions/{session['id']}/turns",
                json={"text": "Please remember this message."},
            )
            assert response.status_code == 503
            messages = client.get(f"/api/v1/sessions/{session['id']}").json()["messages"]
            assert len(messages) == 1
            assert messages[0]["content"] == "Please remember this message."


def test_deleting_session_removes_only_its_linked_memories(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.setenv("TUTOR_NEURAL_MEMORY", "0")
        with TestClient(main.app) as client:
            session = client.post("/api/v1/sessions").json()
            other_session = client.post("/api/v1/sessions").json()
            memory = client.app.state.memory
            linked_id = memory.add("I like hiking.", "You enjoy hiking.", session["id"])
            other_id = memory.add("I like swimming.", "You enjoy swimming.", other_session["id"])
            legacy_id = memory.add("I like reading.", "You enjoy reading.")

            response = client.delete(f"/api/v1/sessions/{session['id']}")

            assert response.status_code == 200
            assert response.json() == {"deleted": True, "memories_deleted": 1}
            assert client.get(f"/api/v1/sessions/{session['id']}").status_code == 404
            remaining_ids = {item["id"] for item in memory.recent(10)}
            assert remaining_ids == {other_id, legacy_id}
            assert linked_id not in remaining_ids


def test_speech_text_normalizes_markdown():
    assert main.clean_text_for_speech("**Hello**, [friend](https://local.test). `Great`!") == "Hello, friend. Great!"
    assert main.clean_text_for_speech("Visit https://example.com now.") == "Visit link omitted now."


def test_speech_text_expands_abbreviations_and_normalizes_symbol_punctuation():
    assert main.clean_text_for_speech("Use **e.g.** apples; pears: fresh, ripe.") == (
        "Use for example apples, pears, fresh, ripe."
    )
    assert main.clean_text_for_speech("Tea & coffee, i.e. both, etc.") == (
        "Tea and coffee, that is both, and so on"
    )
    assert main.clean_text_for_speech("Meet me at 10:30 tomorrow.") == "Meet me at 10:30 tomorrow."


def test_sentence_splitter_keeps_complete_prose_chunks():
    assert main.split_speech_sentences('First thought. “Second thought!”') == [
        "First thought.", '“Second thought!”'
    ]


def test_speech_stream_returns_ordered_bounded_sentence_chunks(monkeypatch):
    import numpy as np

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.setenv("TUTOR_NEURAL_MEMORY", "0")
        monkeypatch.setattr(main, "_get_tts_pipeline", lambda _app: object())
        monkeypatch.setattr(main, "_synthesize", lambda _pipeline, _text: [np.zeros(2400, dtype=np.float32)])
        with TestClient(main.app) as client:
            response = client.post("/api/v1/speech/stream", json={"text": "First sentence. Second sentence."})
            assert response.status_code == 200
            events = [json.loads(line) for line in response.text.splitlines()]
            assert [event["type"] for event in events] == ["audio", "audio", "done"]
            assert [event["text"] for event in events[:2]] == ["First sentence.", "Second sentence."]
            assert [event["index"] for event in events[:2]] == [0, 1]


def test_managed_shutdown_requires_token_and_signals_uvicorn(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.setenv("TUTOR_NEURAL_MEMORY", "0")
        monkeypatch.setenv("TUTOR_SHUTDOWN_TOKEN", "test-only-secret")
        with TestClient(main.app) as client:
            controller = SimpleNamespace(should_exit=False)
            client.app.state.uvicorn_server = controller
            denied = client.post("/api/v1/admin/shutdown", headers={"X-Tutor-Shutdown-Token": "wrong"})
            assert denied.status_code == 403
            assert controller.should_exit is False
            stopped = client.post("/api/v1/admin/shutdown", headers={"X-Tutor-Shutdown-Token": "test-only-secret"})
            assert stopped.status_code == 200
            assert controller.should_exit is True
            del client.app.state.uvicorn_server


def test_websocket_emits_transcript_state_and_reply(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.setenv("TUTOR_NEURAL_MEMORY", "0")
        with TestClient(main.app) as client:
            client.app.state.llm = FakeLLM()
            session = client.post("/api/v1/sessions").json()
            with client.websocket_connect(f"/api/v1/ws/{session['id']}") as socket:
                assert socket.receive_json() == {"type": "state", "state": "idle"}
                socket.send_json({"type": "user_text", "text": "I like to cycle."})
                assert socket.receive_json()["type"] == "transcript"
                assert socket.receive_json() == {"type": "state", "state": "thinking"}
                assert socket.receive_json()["type"] == "assistant_text"
                assert socket.receive_json() == {"type": "state", "state": "speaking"}


def test_websocket_cancel_suppresses_stale_reply_and_recovers(monkeypatch):
    class SlowFirstLLM(FakeLLM):
        def reply(self, messages, memories=None):
            self.calls += 1
            if self.calls == 1:
                time.sleep(0.25)
            return "A fresh response after cancellation."

    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.setenv("TUTOR_NEURAL_MEMORY", "0")
        with TestClient(main.app) as client:
            client.app.state.llm = SlowFirstLLM()
            session = client.post("/api/v1/sessions").json()
            with client.websocket_connect(f"/api/v1/ws/{session['id']}") as socket:
                assert socket.receive_json()["state"] == "idle"
                socket.send_json({"type": "user_text", "text": "Cancel this old turn."})
                assert socket.receive_json()["type"] == "transcript"
                assert socket.receive_json()["state"] == "thinking"
                time.sleep(0.03)
                socket.send_json({"type": "cancel"})
                assert socket.receive_json() == {"type": "state", "state": "interrupted"}
                socket.send_json({"type": "user_text", "text": "Now continue with a new turn."})
                assert socket.receive_json()["type"] == "transcript"
                assert socket.receive_json()["state"] == "thinking"
                answer = socket.receive_json()
                assert answer["assistant_text"] == "A fresh response after cancellation."
                assert socket.receive_json()["state"] == "speaking"


def test_memory_review_api_lists_and_deletes_items(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.setenv("TUTOR_NEURAL_MEMORY", "0")
        with TestClient(main.app) as client:
            memory_id = client.app.state.memory.add("I enjoy hiking.", "You like walking in nature.")
            listed = client.get("/api/v1/memories").json()
            assert listed["memories"][0]["id"] == memory_id
            removed = client.delete(f"/api/v1/memories/{memory_id}")
            assert removed.status_code == 200
            assert client.get("/api/v1/memories").json()["memories"] == []


@pytest.mark.skipif(
    not (main.ROOT / "models" / "embeddings" / "cache" / "models--sentence-transformers--all-MiniLM-L6-v2").exists(),
    reason="local neural embedding weights are not cached",
)
def test_app_startup_uses_cached_neural_embeddings(monkeypatch):
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
        monkeypatch.setattr(main, "DATA", Path(directory))
        monkeypatch.delenv("TUTOR_NEURAL_MEMORY", raising=False)
        with TestClient(main.app) as client:
            assert client.app.state.embedding_status["mode"] == "neural"
            assert client.get("/api/v1/health").json()["models"]["embedding"]["model"] == "sentence-transformers/all-MiniLM-L6-v2"
