from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from retrieval.retriever import RetrievalResult
from scripts.evaluate_retrieval import (
    DEFAULT_GOLD_PATH,
    GoldQuery,
    first_relevant_rank,
    load_gold,
    mean_reciprocal_rank,
    missing_gold_titles,
    recall_at_k,
)


def make_result(title: str, parent_id: str = "") -> RetrievalResult:
    return RetrievalResult(
        id=f"{parent_id or title}:chunk:000",
        parent_id=parent_id or title,
        title=title,
        text="Example text.",
        source="MedlinePlus",
        source_type="health_topic",
        url="https://example.test",
        section="full_summary",
        chunk_index=0,
        distance=0.2,
        similarity=0.8,
    )


class RankingTests(unittest.TestCase):
    def test_first_relevant_rank_matches_titles_case_insensitively(
        self,
    ) -> None:
        results = [
            make_result("Heart Failure"),
            make_result("edema"),
            make_result("Edema"),
        ]
        item = GoldQuery(query="swelling", relevant_titles=["Edema"])

        self.assertEqual(first_relevant_rank(results, item), 2)

    def test_first_relevant_rank_matches_parent_ids(self) -> None:
        results = [
            make_result("Other", parent_id="p1"),
            make_result("Renamed", parent_id="p2"),
        ]
        item = GoldQuery(
            query="q",
            relevant_titles=[],
            relevant_parent_ids=["p2"],
        )

        self.assertEqual(first_relevant_rank(results, item), 2)

    def test_first_relevant_rank_is_none_when_missing(self) -> None:
        item = GoldQuery(query="q", relevant_titles=["Edema"])

        self.assertIsNone(
            first_relevant_rank([make_result("Asthma")], item)
        )


class MetricTests(unittest.TestCase):
    ranks = [1, 3, None, 2]

    def test_recall_at_k(self) -> None:
        self.assertEqual(recall_at_k(self.ranks, 1), 0.25)
        self.assertEqual(recall_at_k(self.ranks, 3), 0.75)
        self.assertEqual(recall_at_k(self.ranks, 5), 0.75)

    def test_mean_reciprocal_rank(self) -> None:
        expected = (1.0 + 1.0 / 3 + 0.0 + 0.5) / 4

        self.assertAlmostEqual(mean_reciprocal_rank(self.ranks), expected)

    def test_empty_ranks_score_zero(self) -> None:
        self.assertEqual(recall_at_k([], 5), 0.0)
        self.assertEqual(mean_reciprocal_rank([]), 0.0)


class GoldFileTests(unittest.TestCase):
    def write_lines(self, lines: list[str]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "gold.jsonl"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def test_loads_gold_queries(self) -> None:
        path = self.write_lines([
            json.dumps({"query": "swollen ankles", "relevant_titles": ["Edema"]}),
            "",
        ])

        gold = load_gold(path)

        self.assertEqual(
            gold,
            [GoldQuery(query="swollen ankles", relevant_titles=["Edema"])],
        )

    def test_rejects_query_without_labels(self) -> None:
        path = self.write_lines([
            json.dumps({"query": "swollen ankles", "relevant_titles": []}),
        ])

        with self.assertRaises(ValueError):
            load_gold(path)

    def test_rejects_unknown_fields(self) -> None:
        path = self.write_lines([
            json.dumps({"query": "q", "relevant_title": ["Edema"]}),
        ])

        with self.assertRaises(ValueError):
            load_gold(path)

    def test_reports_titles_missing_from_corpus(self) -> None:
        corpus = self.write_lines([
            json.dumps({"title": "Edema"}),
            json.dumps({"title": "Asthma"}),
        ])
        gold = [
            GoldQuery(query="a", relevant_titles=["edema", "Edemaa"]),
            GoldQuery(query="b", relevant_titles=["Edemaa", "Asthma"]),
        ]

        self.assertEqual(missing_gold_titles(gold, corpus), ["Edemaa"])

    def test_committed_gold_set_is_valid(self) -> None:
        gold = load_gold(DEFAULT_GOLD_PATH)

        self.assertGreaterEqual(len(gold), 10)


if __name__ == "__main__":
    unittest.main()
