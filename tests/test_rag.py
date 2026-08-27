import os
import tempfile
import unittest

try:
    from rag import build_chunks, chunk_text
except ModuleNotFoundError as exc:
    build_chunks = chunk_text = None
    MISSING_DEPENDENCY = str(exc)
else:
    MISSING_DEPENDENCY = ""


@unittest.skipIf(chunk_text is None, f"optional deployment dependency missing: {MISSING_DEPENDENCY}")
class RagTests(unittest.TestCase):
    def test_chunk_overlap_and_limits(self):
        chunks = chunk_text("ก" * 1000, size=450, overlap=60)
        self.assertTrue(chunks)
        self.assertTrue(all(len(chunk) <= 450 for chunk in chunks))
        self.assertEqual(chunks[0][-60:], chunks[1][:60])

    def test_text_file_has_location(self):
        handle, path = tempfile.mkstemp(suffix=".txt")
        try:
            os.write(handle, "ข้อมูลทดสอบภาษาไทย".encode("utf-8"))
            os.close(handle)
            chunks = build_chunks(path)
            self.assertEqual(chunks[0]["location"], "เนื้อหา")
            self.assertIn("ข้อมูลทดสอบ", chunks[0]["text"])
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
