import tempfile
import unittest
from pathlib import Path

from app.memory import ConversationMemory
from app.retrieval.embeddings import LocalSentenceEmbedder


PROJECT = Path(__file__).resolve().parents[1]
MODEL_CACHE = PROJECT / "models" / "embeddings" / "cache"


@unittest.skipUnless(
    (MODEL_CACHE / "models--sentence-transformers--all-MiniLM-L6-v2").exists(),
    "local MiniLM weights have not been downloaded",
)
class NeuralRetrievalTests(unittest.TestCase):
    def test_sqlite_vec_retrieves_paraphrased_learner_fact(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            database = Path(directory) / "memory.sqlite3"
            previous = ConversationMemory(database)
            target = previous.add(
                "My favourite programming language is Python.",
                "You enjoy Python most.",
            )
            previous.add("I like cooking spicy food on weekends.", "That sounds delicious.")
            previous.add("I work as a primary school teacher.", "You teach young children.")
            embedder = LocalSentenceEmbedder(MODEL_CACHE)
            memory = ConversationMemory(database, embedder=embedder)
            self.assertEqual(memory.reindex_neural(), 3)
            matches = memory.search("Which coding language do I like best?", limit=3)
            self.assertEqual(matches[0]["id"], target)
            self.assertEqual(memory.search("Describe a volcano eruption", limit=3), [])
            self.assertEqual(memory.reindex_neural(), 0)


if __name__ == "__main__":
    unittest.main()
