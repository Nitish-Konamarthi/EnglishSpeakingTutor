"""Run a small, labelled local retrieval check and write JSON metrics."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from app.memory import ConversationMemory
from app.retrieval.embeddings import LocalSentenceEmbedder


FIXTURE = ROOT / "tests" / "fixtures" / "retrieval_eval.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "retrieval_eval.json")
    args = parser.parse_args()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    embedder = LocalSentenceEmbedder(ROOT / "models" / "embeddings" / "cache")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    latencies = []
    hits = 0
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        memory = ConversationMemory(Path(directory) / "eval.sqlite3", embedder=embedder)
        ids = {}
        for item in fixture["memories"]:
            ids[item["id"]] = memory.add(item["text"], item["reply"])
        for item in fixture["queries"]:
            start = time.perf_counter()
            results = memory.search(item["query"], limit=3)
            latencies.append(time.perf_counter() - start)
            hits += int(any(result["id"] == ids[item["relevant"]] for result in results))
    report = {
        "model": embedder.model_id,
        "device": "CPU",
        "index": "sqlite-vec cosine + SQLite FTS5 + reciprocal rank fusion",
        "fixture": str(FIXTURE.relative_to(ROOT)),
        "query_count": len(fixture["queries"]),
        "recall_at_3": hits / len(fixture["queries"]),
        "hits_at_3": hits,
        "mean_query_latency_ms": round(1000 * sum(latencies) / max(1, len(latencies)), 2),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "reranker": {"enabled": False, "reason": "No cross-encoder comparison is justified by this five-query fixture."},
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
