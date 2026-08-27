import math
import unittest

from embedding_service import _normalize, LOCAL_DIMENSION


class EmbeddingServiceTests(unittest.TestCase):
    def test_normalize_returns_unit_vectors(self):
        vectors = _normalize([[3.0, 4.0], [0.0, 0.0]])
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in vectors[0])), 1.0)
        self.assertEqual(vectors[1].tolist(), [0.0, 0.0])

    def test_local_dimension_matches_schema(self):
        self.assertEqual(LOCAL_DIMENSION, 384)


if __name__ == "__main__":
    unittest.main()
