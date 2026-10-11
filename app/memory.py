"""Persistent local memory with hybrid vector and SQLite FTS5 retrieval.

Feature-hash vectors are a dependency-free development baseline. The memory
service boundary lets a planned neural embedder and sqlite-vec backend replace
it without changing the conversation orchestration.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


_TOKEN_RE = re.compile(r"[a-z0-9']+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "can", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "here", "him", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "of", "on", "or", "our", "she",
    "so", "that", "the", "their", "them", "there", "they", "this", "to",
    "up", "us", "was", "we", "were", "what", "when", "where", "which",
    "who", "why", "will", "with", "you", "your",
}


def _features(text: str) -> Counter[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    features: Counter[str] = Counter(
        f"w:{token}" for token in tokens if token not in _STOP_WORDS
    )
    for token in tokens:
        if len(token) >= 4 and token not in _STOP_WORDS:
            padded = f"^{token}$"
            features.update(f"c:{padded[i:i + 3]}" for i in range(len(padded) - 2))
    return features


def embed_text(text: str, dimensions: int = 384) -> list[float]:
    """Create a deterministic sparse feature-hash vector for local retrieval."""
    vector = [0.0] * dimensions
    features = _features(text)
    for feature, count in features.items():
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little")
        index = value % dimensions
        sign = 1.0 if (value >> 8) & 1 else -1.0
        vector[index] += sign * (1.0 + math.log(count))
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class ConversationMemory:
    """SQLite-backed hybrid index for durable conversation episodes.

    FTS5 provides exact-term matching; the vector column is behind this small
    interface so a neural embedding model and sqlite-vec can replace the
    dependency-free feature-hash baseline without changing the tutor loop.
    """

    def __init__(self, database_path: Path | str, dimensions: int = 384):
        self.database_path = str(database_path)
        self.dimensions = dimensions
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    created_at REAL NOT NULL
                )"""
            )
            connection.execute(
                """CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                    user_text, assistant_text,
                    content='memories', content_rowid='id'
                )"""
            )
            connection.executescript(
                """
                CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                    INSERT INTO memories_fts(rowid, user_text, assistant_text)
                    VALUES (new.id, new.user_text, new.assistant_text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, user_text, assistant_text)
                    VALUES ('delete', old.id, old.user_text, old.assistant_text);
                END;
                CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, user_text, assistant_text)
                    VALUES ('delete', old.id, old.user_text, old.assistant_text);
                    INSERT INTO memories_fts(rowid, user_text, assistant_text)
                    VALUES (new.id, new.user_text, new.assistant_text);
                END;
                """
            )
            # Rebuild once at startup so rows created by the earlier schema
            # version are searchable through FTS5 as well.
            connection.execute("INSERT INTO memories_fts(memories_fts) VALUES ('rebuild')")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def add(self, user_text: str, assistant_text: str) -> int:
        """Persist one completed tutor exchange and its search vector."""
        text = f"Learner: {user_text}\nTutor: {assistant_text}"
        vector = embed_text(text, self.dimensions)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memories(user_text, assistant_text, embedding, created_at) "
                "VALUES (?, ?, ?, ?)",
                (user_text, assistant_text, json.dumps(vector), time.time()),
            )
            return int(cursor.lastrowid)

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Fuse vector and exact-term candidates, then apply recency scoring."""
        query_vector = embed_text(query, self.dimensions)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, user_text, assistant_text, embedding, created_at FROM memories"
            ).fetchall()
            terms = [
                term.lower() for term in _TOKEN_RE.findall(query)
                if term.lower() not in _STOP_WORDS and len(term) > 1
            ]
            fts_rows = []
            if terms:
                fts_query = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
                fts_rows = connection.execute(
                    "SELECT rowid AS id FROM memories_fts "
                    "WHERE memories_fts MATCH ? ORDER BY bm25(memories_fts) LIMIT 20",
                    (fts_query,),
                ).fetchall()

        vector_matches = []
        for row in rows:
            vector = json.loads(row["embedding"])
            score = cosine_similarity(query_vector, vector)
            vector_matches.append({
                "id": row["id"],
                "user_text": row["user_text"],
                "assistant_text": row["assistant_text"],
                "created_at": row["created_at"],
                "similarity": score,
            })
        vector_matches.sort(key=lambda item: item["similarity"], reverse=True)
        vector_matches = [
            item for item in vector_matches if item["similarity"] >= 0.20
        ]
        vector_rank = {item["id"]: rank for rank, item in enumerate(vector_matches[:20], 1)}
        lexical_rank = {row["id"]: rank for rank, row in enumerate(fts_rows, 1)}
        by_id = {item["id"]: item for item in vector_matches}
        candidate_ids = set(vector_rank) | set(lexical_rank)
        if not candidate_ids:
            return []

        now = time.time()
        scored = []
        for item_id in candidate_ids:
            item = by_id[item_id]
            # Reciprocal rank fusion puts vector and BM25 candidates on a
            # comparable scale before recency/importance are applied.
            rrf = (1 / (60 + vector_rank[item_id]) if item_id in vector_rank else 0)
            rrf += (1 / (60 + lexical_rank[item_id]) if item_id in lexical_rank else 0)
            recency = math.exp(-math.log(2) * max(0, now - item["created_at"]) / (14 * 86400))
            dense_score = max(0.0, item["similarity"])
            item["score"] = 0.6 * dense_score + 0.25 * recency + 0.15 * 0.5 + rrf
            scored.append(item)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:max(0, limit)]
