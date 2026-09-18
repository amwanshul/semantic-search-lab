import unittest

from app import TfidfSearch, DOCUMENTS


class TestSearch(unittest.TestCase):
    def test_relevant_document_ranks(self):
        engine = TfidfSearch(DOCUMENTS)
        results = engine.search("machine learning", top_k=3)
        self.assertIn("machine learning", results[0]["document"].lower())

    def test_unknown_query_returns_zero_scores(self):
        engine = TfidfSearch(DOCUMENTS)
        results = engine.search("quantum banana", top_k=2)
        self.assertEqual(results[0]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
