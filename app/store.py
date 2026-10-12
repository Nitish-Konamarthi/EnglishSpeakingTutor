"""Small SQLite store for durable tutoring sessions and turns."""

from __future__ import annotations

import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class SessionStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, title TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL, client_turn_id TEXT,
                    created_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS messages_client_turn
                    ON messages(session_id, client_turn_id, role)
                    WHERE client_turn_id IS NOT NULL;
                CREATE INDEX IF NOT EXISTS messages_session_order
                    ON messages(session_id, id);
            """)

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            yield db
            db.commit()
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    def create_session(self, title: str = "New conversation") -> dict[str, Any]:
        session_id, now = str(uuid.uuid4()), time.time()
        with self._connect() as db:
            db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", (session_id, title, now, now))
        return {"id": session_id, "title": title, "created_at": now, "updated_at": now}

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    def messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        with self._connect() as db:
            if limit is None:
                rows = db.execute("SELECT id, role, content, client_turn_id, created_at FROM messages WHERE session_id=? ORDER BY id", (session_id,)).fetchall()
            else:
                rows = db.execute("SELECT id, role, content, client_turn_id, created_at FROM (SELECT * FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?) ORDER BY id", (session_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def add_message(self, session_id: str, role: str, content: str, client_turn_id: str | None = None) -> dict[str, Any]:
        now = time.time()
        with self._connect() as db:
            cursor = db.execute(
                "INSERT OR IGNORE INTO messages(session_id,role,content,client_turn_id,created_at) VALUES (?,?,?,?,?)",
                (session_id, role, content, client_turn_id, now),
            )
            if cursor.rowcount == 0 and client_turn_id:
                row = db.execute("SELECT id, role, content, client_turn_id, created_at FROM messages WHERE session_id=? AND client_turn_id=? AND role=?", (session_id, client_turn_id, role)).fetchone()
                return dict(row)
            db.execute("UPDATE sessions SET updated_at=?, title=CASE WHEN title='New conversation' AND ?='user' THEN substr(?,1,64) ELSE title END WHERE id=?", (now, role, content, session_id))
            row = db.execute("SELECT id, role, content, client_turn_id, created_at FROM messages WHERE id=?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def turn_messages(self, session_id: str, client_turn_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT id,role,content,client_turn_id,created_at FROM messages WHERE session_id=? AND client_turn_id=? ORDER BY id",
                (session_id, client_turn_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as db:
            cursor = db.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        return cursor.rowcount > 0

