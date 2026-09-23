import unittest
from pathlib import Path

from app.adapters.arxiv_parser import parse_feed, total_results

FIXTURE = Path(__file__).parent / "fixtures" / "sample_arxiv_feed.xml"


class TestArxivParser(unittest.TestCase):
    def setUp(self):
        self.xml_bytes = FIXTURE.read_bytes()
        self.records = parse_feed(self.xml_bytes)

    def test_total_results(self):
        self.assertEqual(total_results(self.xml_bytes), 2)

    def test_parses_both_entries(self):
        self.assertEqual(len(self.records), 2)

    def test_modern_paper_full_fields(self):
        paper = self.records[0]
        self.assertEqual(paper.arxiv_id, "2401.01234")
        self.assertEqual(paper.version, 2)
        # Title newline should be collapsed to a single space, no double spaces.
        self.assertEqual(
            paper.title, "Retrieval-Augmented Generation for Scientific Question Answering"
        )
        self.assertIn("hallucination", paper.abstract)
        self.assertNotIn("\n", paper.abstract)
        self.assertEqual(paper.authors, ["Jane Doe", "John Roe"])
        self.assertEqual(paper.primary_category, "cs.CL")
        self.assertEqual(set(paper.categories), {"cs.CL", "cs.AI"})
        self.assertEqual(paper.doi, "10.1000/xyz123")
        self.assertEqual(paper.journal_ref, "Journal of Made Up Science, 2024")
        self.assertEqual(paper.url, "http://arxiv.org/abs/2401.01234v2")
        self.assertEqual(paper.published_at.year, 2024)
        self.assertTrue(len(paper.content_hash) == 64)  # sha256 hex digest

    def test_old_style_id_and_missing_optional_fields(self):
        paper = self.records[1]
        # Old-style ids contain a slash and must be preserved verbatim (no version stripped wrong).
        self.assertEqual(paper.arxiv_id, "hep-th/9901001")
        self.assertEqual(paper.version, 1)
        self.assertIsNone(paper.doi)
        self.assertIsNone(paper.journal_ref)
        self.assertEqual(paper.primary_category, "hep-th")
        self.assertEqual(paper.categories, ["hep-th"])

    def test_content_hash_is_deterministic_and_sensitive_to_abstract(self):
        from app.adapters.text_cleaner import content_hash

        h1 = content_hash("Title", "Abstract A")
        h2 = content_hash("Title", "Abstract A")
        h3 = content_hash("Title", "Abstract B")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)

    def test_malformed_entry_is_skipped_not_fatal(self):
        broken_xml = b"""<?xml version="1.0"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry><title>No id or summary here</title></entry>
        </feed>"""
        records = parse_feed(broken_xml)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
