import tempfile
import unittest
from pathlib import Path

from app.memory import ConversationMemory, cosine_similarity, embed_text


class ConversationMemoryTests(unittest.TestCase):
    def test_feature_vectors_are_normalized_and_repeatable(self):
        first = embed_text("I enjoy learning Python")
        second = embed_text("I enjoy learning Python")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 384)
        self.assertAlmostEqual(cosine_similarity(first, first), 1.0)

    def test_hybrid_search_finds_and_persists_an_exact_memory(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            database = Path(directory) / "memory.sqlite3"
            memory = ConversationMemory(database)
            memory.add(
                "My favourite programming language is Python.",
                "You enjoy using Python.",
            )

            result = memory.search(
                "What is my favourite programming language?", limit=3
            )

            self.assertEqual(len(result), 1)
            self.assertTrue(result[0]["user_text"].endswith("Python."))
            self.assertGreater(result[0]["score"], 0)

            reopened = ConversationMemory(database)
            persisted = reopened.search("Python programming language", limit=3)
            self.assertEqual(persisted[0]["id"], result[0]["id"])

    def test_unrelated_query_does_not_return_random_hash_matches(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            memory = ConversationMemory(Path(directory) / "memory.sqlite3")
            memory.add("I like Python automation.", "That sounds useful.")

            self.assertEqual(memory.search("Describe a volcano eruption", limit=3), [])


if __name__ == "__main__":
    unittest.main()
