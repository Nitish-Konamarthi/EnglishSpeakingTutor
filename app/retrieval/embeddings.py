"""CPU-only local neural sentence embeddings with an explicit offline cache."""

from __future__ import annotations

import os
import threading
from pathlib import Path


class LocalSentenceEmbedder:
    def __init__(self, cache_dir: str | Path, model_id: str | None = None):
        self.model_id = model_id or os.environ.get(
            "TUTOR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.cache_dir = Path(cache_dir)
        self.model = None
        self._lock = threading.Lock()
        self._encode_lock = threading.Lock()

    def load(self):
        if self.model is not None:
            return self.model
        with self._lock:
            if self.model is None:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(
                    self.model_id,
                    device="cpu",
                    cache_folder=str(self.cache_dir),
                    local_files_only=True,
                )
        return self.model

    def encode(self, texts: list[str]) -> list[list[float]]:
        with self._encode_lock:
            vectors = self.load().encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return vectors.tolist()

    @property
    def dimensions(self) -> int:
        model = self.load()
        getter = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        return int(getter())
