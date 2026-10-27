from __future__ import annotations

import unittest

from retrieval.chunking import DEFAULT_CHUNKING_CONFIG
from retrieval.schema import RetrievalChunk
from retrieval.text_utils import parse_summary_html
from scripts.build_retrieval_corpus import validate_chunks


def make_chunk(**overrides: object) -> RetrievalChunk:
    values: dict[str, object] = {
        "id": "chunk-1",
        "parent_id": "parent-1",
        "title": "Example",
        "text": "Plain text summary.",
        "source": "MedlinePlus",
        "source_type": "health_topic",
        "url": "https://example.test/topic",
        "section": "full_summary",
        "chunk_index": 0,
    }
    values.update(overrides)
    return RetrievalChunk(**values)  # type: ignore[arg-type]


class SummaryParsingTests(unittest.TestCase):
    def test_preserves_real_h2_h3_and_h4_headings(self) -> None:
        html = """
            <p>Overview.</p>
            <h2>Symptoms</h2><p>Symptom details.</p>
            <h3>Diagnosis</h3><p>Diagnosis details.</p>
            <h4>Treatment</h4><p>Treatment details.</p>
        """

        sections = parse_summary_html(html)

        self.assertEqual(
            [name for name, _ in sections],
            ["full_summary", "Symptoms", "Diagnosis", "Treatment"],
        )

    def test_uses_full_summary_when_there_are_no_headings(self) -> None:
        sections = parse_summary_html(
            "<p>First paragraph.</p><p>Second paragraph.</p>"
        )

        self.assertEqual(
            sections,
            [("full_summary", ["First paragraph.", "Second paragraph."])],
        )


class CorpusValidationTests(unittest.TestCase):
    def assert_invalid(self, chunks: list[RetrievalChunk]) -> None:
        with self.assertRaises(ValueError):
            validate_chunks(chunks)

    def test_accepts_valid_chunks(self) -> None:
        validate_chunks([make_chunk()])

    def test_rejects_duplicate_ids(self) -> None:
        self.assert_invalid([make_chunk(), make_chunk()])

    def test_rejects_empty_text(self) -> None:
        self.assert_invalid([make_chunk(text="  ")])

    def test_rejects_dirty_glossary_prefixes(self) -> None:
        self.assert_invalid([
            make_chunk(
                source_type="glossary_definition",
                title=">Term",
            )
        ])
        self.assert_invalid([
            make_chunk(
                source_type="glossary_definition",
                text=">Definition",
            )
        ])

    def test_rejects_raw_html(self) -> None:
        self.assert_invalid([make_chunk(text="A <strong>raw</strong> tag.")])

    def test_rejects_empty_sections(self) -> None:
        self.assert_invalid([make_chunk(section="")])

    def test_rejects_chunks_over_the_configured_limit(self) -> None:
        self.assert_invalid([
            make_chunk(
                text="x" * (DEFAULT_CHUNKING_CONFIG.max_chars + 1)
            )
        ])


if __name__ == "__main__":
    unittest.main()
