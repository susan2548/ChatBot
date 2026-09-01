import unittest
import sys
import types
from unittest import mock


def _import_db_with_lightweight_dependency_stubs():
    """Import chat persistence code without installing deployment packages."""
    names = (
        "psycopg2", "psycopg2.extras", "psycopg2.pool",
        "pgvector", "pgvector.psycopg2", "google", "google.genai", "rag",
    )
    previous = {name: sys.modules.get(name) for name in names}

    psycopg2_module = types.ModuleType("psycopg2")
    extras_module = types.ModuleType("psycopg2.extras")
    extras_module.execute_batch = lambda *args, **kwargs: None
    extras_module.Json = lambda value: value
    pool_module = types.ModuleType("psycopg2.pool")
    pool_module.ThreadedConnectionPool = object
    psycopg2_module.extras = extras_module
    psycopg2_module.pool = pool_module
    psycopg2_module.connect = lambda *args, **kwargs: None

    pgvector_module = types.ModuleType("pgvector")
    pgvector_module.Vector = lambda value: value
    pgvector_psycopg2_module = types.ModuleType("pgvector.psycopg2")
    pgvector_psycopg2_module.register_vector = lambda connection: None

    google_module = types.ModuleType("google")
    genai_module = types.ModuleType("google.genai")
    genai_module.Client = object
    genai_module.types = types.SimpleNamespace()
    google_module.genai = genai_module

    rag_module = types.ModuleType("rag")
    rag_module.read_file = lambda path: ""
    rag_module.chunk_text = lambda text, size=450, overlap=60: []
    rag_module.build_chunks = lambda path: []
    rag_module.iter_chunks = lambda path: iter(())
    rag_module.SUPPORTED_EXTS = set()
    rag_module.ZIP_EXT = ".zip"

    sys.modules.update({
        "psycopg2": psycopg2_module,
        "psycopg2.extras": extras_module,
        "psycopg2.pool": pool_module,
        "pgvector": pgvector_module,
        "pgvector.psycopg2": pgvector_psycopg2_module,
        "google": google_module,
        "google.genai": genai_module,
        "rag": rag_module,
    })
    try:
        import db as imported_db
        return imported_db
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


db = _import_db_with_lightweight_dependency_stubs()


class FakeCursor:
    def __init__(self, fetchall_rows=None, owner_id=7, next_seq=4):
        self.fetchall_rows = fetchall_rows or []
        self.owner_id = owner_id
        self.next_seq = next_seq
        self.executed = []
        self._last_query = ""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, query, params=None):
        self._last_query = query
        self.executed.append((query, params))

    def fetchall(self):
        return self.fetchall_rows

    def fetchone(self):
        if "MAX(seq)" in self._last_query:
            return (self.next_seq,)
        if "SELECT owner_id" in self._last_query:
            return (self.owner_id,)
        return None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


class FakeConnectionContext:
    def __init__(self, connection):
        self.connection = connection

    def __enter__(self):
        return self.connection

    def __exit__(self, exc_type, exc, tb):
        return False


class ChatDatabaseTests(unittest.TestCase):
    def test_list_chat_summaries_uses_one_query_and_maps_all_sidebar_fields(self):
        cursor = FakeCursor([
            ("chat_a.json", "เรื่อง C", True, "c-language", "ภาษา C"),
            ("chat_b.json", None, False, None, None),
        ])
        connection = FakeConnection(cursor)
        with mock.patch.object(
            db, "_get_conn", return_value=FakeConnectionContext(connection)
        ):
            summaries = db.list_chat_summaries(owner_id=7)

        self.assertEqual(len(cursor.executed), 1)
        self.assertEqual(summaries[0]["title"], "เรื่อง C")
        self.assertTrue(summaries[0]["pinned"])
        self.assertEqual(summaries[0]["topic_name"], "ภาษา C")

    def test_append_messages_checks_owner_and_allocates_consecutive_sequences(self):
        cursor = FakeCursor(owner_id=7, next_seq=4)
        connection = FakeConnection(cursor)
        captured_rows = []

        def capture_batch(cur, query, rows, page_size=None):
            captured_rows.extend(rows)

        with mock.patch.object(
            db, "_get_conn", return_value=FakeConnectionContext(connection)
        ), mock.patch.object(db.psycopg2.extras, "execute_batch", side_effect=capture_batch):
            count = db.append_chat_messages(
                "chat_a.json",
                [
                    {"role": "user", "content": "ถาม", "timestamp": "เวลา"},
                    {"role": "model", "content": "ตอบ", "timestamp": "เวลา"},
                ],
                owner_id=7,
            )

        self.assertEqual(count, 2)
        self.assertEqual([row[-1] for row in captured_rows], [4, 5])
        self.assertTrue(connection.committed)

    def test_append_rejects_a_different_owner(self):
        cursor = FakeCursor(owner_id=99)
        connection = FakeConnection(cursor)
        with mock.patch.object(
            db, "_get_conn", return_value=FakeConnectionContext(connection)
        ):
            with self.assertRaises(PermissionError):
                db.append_chat_messages(
                    "chat_a.json",
                    [{"role": "user", "content": "ห้ามข้ามบัญชี"}],
                    owner_id=7,
                )

    def test_load_chat_page_returns_chronological_window_and_more_flag(self):
        cursor = FakeCursor([
            ("model", "ล่าสุด", "t3", 3),
            ("user", "ก่อนหน้า", "t2", 2),
            ("model", "เก่ากว่า", "t1", 1),
        ])
        connection = FakeConnection(cursor)
        with mock.patch.object(
            db, "_get_conn", return_value=FakeConnectionContext(connection)
        ):
            messages, has_more, oldest_seq = db.load_chat_page(
                "chat_a.json", owner_id=7, limit=2
            )

        self.assertTrue(has_more)
        self.assertEqual([item["content"] for item in messages], ["ก่อนหน้า", "ล่าสุด"])
        self.assertEqual(oldest_seq, 2)

    def test_cached_answer_keeps_knowledge_sources(self):
        cursor = FakeCursor()
        cursor.fetchone = lambda: (
            "คำตอบ",
            [{"source": "c.csv", "location": "แถว 10"}],
        )
        connection = FakeConnection(cursor)
        with mock.patch.object(
            db, "_get_conn", return_value=FakeConnectionContext(connection)
        ):
            entry = db.get_cached_answer("cache-key")

        self.assertEqual(entry["answer"], "คำตอบ")
        self.assertEqual(entry["sources"][0]["source"], "c.csv")

    def test_hybrid_retrieval_uses_one_database_execute(self):
        cursor = FakeCursor([
            (
                "หัวข้อ: printf | เนื้อหา: ใช้ printf แสดงผล",
                "c.csv",
                {"location": "แถว 2"},
                0.9,
            )
        ])
        connection = FakeConnection(cursor)
        provider = types.SimpleNamespace(
            profile="local-minilm-l12-v1",
            embed_query=lambda text: [0.1, 0.2],
        )
        with mock.patch.object(
            db, "_get_vector_conn", return_value=FakeConnectionContext(connection)
        ), mock.patch.object(db, "get_embedding_provider", return_value=provider):
            results = db.search_with_sources("c-language", "printf คืออะไร", top_k=5)

        self.assertEqual(len(cursor.executed), 1)
        self.assertEqual(results[0]["source"], "c.csv")


if __name__ == "__main__":
    unittest.main()
