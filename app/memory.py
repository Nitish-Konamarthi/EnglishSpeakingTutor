"""Persistent local memory with hybrid vector and SQLite FTS5 retrieval.

Feature-hash vectors remain available as a dependency-free test/offline
fallback. The browser application injects a local neural embedder and uses
sqlite-vec for dense candidates plus FTS5 for lexical candidates.
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

    def __init__(self, database_path: Path | str, dimensions: int = 384, embedder=None):
        self.database_path = str(database_path)
        self.embedder = embedder
        self.dimensions = int(embedder.dimensions) if embedder else dimensions
        self.embedding_model = embedder.model_id if embedder else "feature-hash-v1"
        self.vector_index = embedder is not None
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_text TEXT NOT NULL,
                    assistant_text TEXT NOT NULL,
                    embedding TEXT NOT NULL,
                    embedding_model TEXT NOT NULL DEFAULT 'feature-hash-v1',
                    created_at REAL NOT NULL
                )"""
            )
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(memories)")}
            if "embedding_model" not in columns:
                connection.execute("ALTER TABLE memories ADD COLUMN embedding_model TEXT NOT NULL DEFAULT 'feature-hash-v1'")
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
            if self.vector_index:
                connection.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec USING vec0(memory_id INTEGER PRIMARY KEY, embedding FLOAT[{self.dimensions}] distance_metric=cosine)"
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            try:
                # A database may already contain memories_vec even when the
                # current startup must use the lexical fallback.
                import sqlite_vec
                connection.enable_load_extension(True)
                sqlite_vec.load(connection)
                connection.enable_load_extension(False)
            except ModuleNotFoundError:
                if self.vector_index:
                    raise
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
        vector = self.embedder.encode([text])[0] if self.embedder else embed_text(text, self.dimensions)
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO memories(user_text, assistant_text, embedding, embedding_model, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (user_text, assistant_text, json.dumps(vector), self.embedding_model, time.time()),
            )
            memory_id = int(cursor.lastrowid)
            if self.vector_index:
                import sqlite_vec
                connection.execute(
                    "INSERT INTO memories_vec(memory_id, embedding) VALUES (?, ?)",
                    (memory_id, sqlite_vec.serialize_float32(vector)),
                )
            return memory_id

    def search(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Fuse vector and exact-term candidates, then apply recency scoring."""
        query_vector = self.embedder.encode([query])[0] if self.embedder else embed_text(query, self.dimensions)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, user_text, assistant_text, embedding, embedding_model, created_at FROM memories"
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

            vector_rows = []
            if self.vector_index:
                import sqlite_vec
                vector_rows = connection.execute(
                    "SELECT memory_id, distance FROM memories_vec WHERE embedding MATCH ? AND k = ?",
                    (sqlite_vec.serialize_float32(query_vector), 20),
                ).fetchall()

        by_id = {
            row["id"]: {
                "id": row["id"], "user_text": row["user_text"],
                "assistant_text": row["assistant_text"], "created_at": row["created_at"],
                "similarity": 0.0, "embedding_model": row["embedding_model"],
            }
            for row in rows
        }
        if self.vector_index:
            vector_matches = []
            for row in vector_rows:
                item = by_id.get(row["memory_id"])
                if item and item["embedding_model"] == self.embedding_model:
                    item["similarity"] = max(-1.0, 1.0 - float(row["distance"]))
                    if item["similarity"] >= 0.25:
                        vector_matches.append(item)
        else:
            vector_matches = []
            for row in rows:
                if row["embedding_model"] != self.embedding_model:
                    continue
                item = by_id[row["id"]]
                item["similarity"] = cosine_similarity(query_vector, json.loads(row["embedding"]))
                vector_matches.append(item)
            vector_matches.sort(key=lambda item: item["similarity"], reverse=True)
            vector_matches = [item for item in vector_matches if item["similarity"] >= 0.20]
        vector_rank = {item["id"]: rank for rank, item in enumerate(vector_matches[:20], 1)}
        lexical_rank = {row["id"]: rank for rank, row in enumerate(fts_rows, 1)}
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

    def reindex_neural(self, batch_size: int = 32) -> int:
        """Backfill neural vectors for old turns without blocking live requests."""
        if not self.vector_index:
            return 0
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,user_text,assistant_text FROM memories WHERE embedding_model != ? ORDER BY id",
                (self.embedding_model,),
            ).fetchall()
        count = 0
        for offset in range(0, len(rows), max(1, batch_size)):
            batch = rows[offset:offset + batch_size]
            texts = [f"Learner: {row['user_text']}\nTutor: {row['assistant_text']}" for row in batch]
            vectors = self.embedder.encode(texts)
            with self._connect() as connection:
                import sqlite_vec
                for row, vector in zip(batch, vectors):
                    connection.execute(
                        "UPDATE memories SET embedding=?,embedding_model=? WHERE id=?",
                        (json.dumps(vector), self.embedding_model, row["id"]),
                    )
                    connection.execute("DELETE FROM memories_vec WHERE memory_id=?", (row["id"],))
                    connection.execute(
                        "INSERT INTO memories_vec(memory_id,embedding) VALUES (?,?)",
                        (row["id"], sqlite_vec.serialize_float32(vector)),
                    )
                    count += 1
        return count

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,user_text,assistant_text,embedding_model,created_at FROM memories ORDER BY id DESC LIMIT ?",
                (max(0, min(limit, 200)),),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete(self, memory_id: int) -> bool:
        with self._connect() as connection:
            if self.vector_index:
                connection.execute("DELETE FROM memories_vec WHERE memory_id=?", (memory_id,))
            cursor = connection.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        return cursor.rowcount > 0
