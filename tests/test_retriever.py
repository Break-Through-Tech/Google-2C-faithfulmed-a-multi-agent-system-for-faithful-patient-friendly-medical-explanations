from __future__ import annotations

import unittest

from retrieval.retriever import RetrievalResult, Retriever
from tests.test_index import make_chunk, make_index


class FakeQueryEmbedder:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [1.0, 0.0, 0.0]
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return self.vector


EDEMA = make_chunk(
    id="medlineplus:health_topic:1:chunk:000",
    parent_id="medlineplus:health_topic:1",
    title="Edema",
    text="Edema is swelling caused by fluid in your body's tissues.",
    url="https://medlineplus.gov/edema.html",
    section="full_summary",
    chunk_index=0,
    synonyms=[],
)

HEART_FAILURE = make_chunk(
    id="medlineplus:health_topic:2:chunk:001",
    parent_id="medlineplus:health_topic:2",
    title="Heart Failure",
    text="Heart failure can cause swelling in the legs.",
    url="https://medlineplus.gov/heartfailure.html",
    section="Symptoms",
    chunk_index=1,
)

VITAMIN_D = make_chunk(
    id="medlineplus:glossary:vitamins:vitamin-d:chunk:000",
    parent_id="medlineplus:glossary:vitamins:vitamin-d",
    title="Vitamin D",
    text="Vitamin D helps your body absorb calcium.",
    source_type="glossary_definition",
    url="https://medlineplus.gov/definitions/vitaminsdefinitions.html",
    section="definition",
    chunk_index=0,
)


class RetrieverTests(unittest.TestCase):
    def make_retriever(
        self,
        embedder: FakeQueryEmbedder | None = None,
        *,
        empty: bool = False,
    ) -> tuple[Retriever, FakeQueryEmbedder]:
        index = make_index()

        if not empty:
            index.upsert(
                [EDEMA, HEART_FAILURE, VITAMIN_D],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            )

        embedder = embedder or FakeQueryEmbedder()
        return Retriever(index, embedder), embedder

    def test_returns_text_and_provenance(self) -> None:
        retriever, embedder = self.make_retriever()

        results = retriever.retrieve("bilateral lower-extremity edema", k=2)

        self.assertEqual(embedder.queries, ["bilateral lower-extremity edema"])
        self.assertEqual(
            [result.title for result in results],
            ["Edema", "Heart Failure"],
        )

        top = results[0]
        self.assertEqual(top.id, EDEMA.id)
        self.assertEqual(top.parent_id, EDEMA.parent_id)
        self.assertEqual(top.text, EDEMA.text)
        self.assertEqual(top.source, "MedlinePlus")
        self.assertEqual(top.source_type, "health_topic")
        self.assertEqual(top.url, "https://medlineplus.gov/edema.html")
        self.assertEqual(top.section, "full_summary")
        self.assertEqual(top.chunk_index, 0)
        self.assertEqual(results[1].section, "Symptoms")

    def test_similarity_is_cosine_and_distance_is_its_complement(
        self,
    ) -> None:
        retriever, _ = self.make_retriever()

        top, second = retriever.retrieve("edema", k=2)

        self.assertAlmostEqual(top.similarity, 1.0, places=5)
        self.assertAlmostEqual(second.similarity, 2 ** -0.5, places=5)
        self.assertAlmostEqual(second.distance, 1 - 2 ** -0.5, places=5)

    def test_respects_k(self) -> None:
        retriever, _ = self.make_retriever()

        self.assertEqual(len(retriever.retrieve("edema", k=1)), 1)

    def test_filters_by_source_type(self) -> None:
        retriever, _ = self.make_retriever()

        results = retriever.retrieve(
            "vitamin D",
            k=5,
            source_type="glossary_definition",
        )

        self.assertEqual([result.title for result in results], ["Vitamin D"])

    def test_empty_collection_returns_no_results(self) -> None:
        retriever, _ = self.make_retriever(empty=True)

        self.assertEqual(retriever.retrieve("edema"), [])

    def test_rejects_empty_query(self) -> None:
        retriever, embedder = self.make_retriever()

        with self.assertRaises(ValueError):
            retriever.retrieve("   ")

        self.assertEqual(embedder.queries, [])

    def test_rejects_non_positive_k(self) -> None:
        retriever, _ = self.make_retriever()

        with self.assertRaises(ValueError):
            retriever.retrieve("edema", k=0)

    def test_rejects_wrong_query_vector_dimension(self) -> None:
        retriever, _ = self.make_retriever(
            FakeQueryEmbedder([1.0, 0.0, 0.0, 0.0])
        )

        with self.assertRaisesRegex(ValueError, "expected 3"):
            retriever.retrieve("edema")

    def test_result_serializes_to_dict(self) -> None:
        retriever, _ = self.make_retriever()

        result = retriever.retrieve("edema", k=1)[0]

        self.assertIsInstance(result, RetrievalResult)
        self.assertEqual(result.to_dict()["title"], "Edema")
        self.assertEqual(
            result.to_dict()["url"],
            "https://medlineplus.gov/edema.html",
        )


if __name__ == "__main__":
    unittest.main()
