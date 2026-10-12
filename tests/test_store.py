import tempfile
import unittest
from pathlib import Path

from app.store import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_messages_survive_reopening_and_are_isolated_by_session(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "sessions.sqlite3"
            store = SessionStore(path)
            first = store.create_session()
            second = store.create_session()
            store.add_message(first["id"], "user", "My hobby is cycling.", "turn-1")
            reopened = SessionStore(path)
            self.assertEqual(reopened.messages(first["id"])[0]["content"], "My hobby is cycling.")
            self.assertEqual(reopened.messages(second["id"]), [])

    def test_client_turn_id_prevents_duplicate_message(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            store = SessionStore(Path(directory) / "sessions.sqlite3")
            session = store.create_session()
            first = store.add_message(session["id"], "user", "Hello", "same-id")
            duplicate = store.add_message(session["id"], "user", "Hello", "same-id")
            self.assertEqual(first["id"], duplicate["id"])
            self.assertEqual(len(store.messages(session["id"])), 1)


if __name__ == "__main__":
    unittest.main()
