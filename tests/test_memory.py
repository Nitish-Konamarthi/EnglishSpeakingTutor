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

    def test_memory_review_and_delete(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            memory = ConversationMemory(Path(directory) / "memory.sqlite3")
            item_id = memory.add("I enjoy cycling.", "That sounds fun.")
            self.assertEqual(memory.recent(5)[0]["id"], item_id)
            self.assertTrue(memory.delete(item_id))
            self.assertEqual(memory.recent(5), [])
            self.assertFalse(memory.delete(item_id))

    def test_delete_session_memories_preserves_other_and_legacy_memories(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            memory = ConversationMemory(Path(directory) / "memory.sqlite3")
            owned_id = memory.add("I like hiking.", "You enjoy hiking.", "session-one")
            other_id = memory.add("I like swimming.", "You enjoy swimming.", "session-two")
            legacy_id = memory.add("I like reading.", "You enjoy reading.")

            self.assertEqual(memory.delete_for_session("session-one"), 1)
            remaining_ids = {item["id"] for item in memory.recent(10)}
            self.assertNotIn(owned_id, remaining_ids)
            self.assertEqual(remaining_ids, {other_id, legacy_id})


if __name__ == "__main__":
    unittest.main()
