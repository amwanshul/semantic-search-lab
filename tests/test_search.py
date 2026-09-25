import unittest

from app import DOCUMENTS, SearchRequest, TfidfSearch


class TestSearch(unittest.TestCase):
    def test_search_request_rejects_non_positive_top_k(self):
        with self.assertRaises(Exception):
            SearchRequest(query="python", top_k=0)

    def test_relevant_document_ranks_first(self):
        engine = TfidfSearch(DOCUMENTS)
        results = engine.search("machine learning", top_k=3)
        self.assertIn("machine learning", results[0]["document"].lower())

    def test_unknown_query_returns_zero_scores(self):
        engine = TfidfSearch(DOCUMENTS)
        results = engine.search("quantum banana", top_k=2)
        self.assertEqual(results[0]["score"], 0.0)

    def test_large_top_k_is_bounded_by_document_count(self):
        engine = TfidfSearch(DOCUMENTS)
        results = engine.search("python", top_k=100)
        self.assertEqual(len(results), len(DOCUMENTS))


if __name__ == "__main__":
    unittest.main()
