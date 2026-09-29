from __future__ import annotations

import unittest
from typing import Any

from retrieval.embeddings import EmbeddingConfig
from retrieval.index import ChromaCloudIndex
from retrieval.retriever import RetrievalResult, Retriever
from tests.test_index import FakeChromaClient, FakeCollection


class QueryableCollection(FakeCollection):
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        super().__init__(dimensions=3)
        self.hits = hits
        self.query_calls: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
        self.query_calls.append(kwargs)
        hits = self.hits[:kwargs["n_results"]]
        return {
            "ids": [[hit["id"] for hit in hits]],
            "documents": [[hit["document"] for hit in hits]],
            "metadatas": [[hit["metadata"] for hit in hits]],
            "distances": [[hit["distance"] for hit in hits]],
        }


class FakeQueryEmbedder:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.queries: list[str] = []

    def embed_query(self, query: str) -> list[float]:
        self.queries.append(query)
        return [0.1] * self.dimensions


EDEMA_HIT = {
    "id": "medlineplus:health_topic:1:chunk:000",
    "document": "Edema is swelling caused by fluid in your body's tissues.",
    "metadata": {
        "parent_id": "medlineplus:health_topic:1",
        "title": "Edema",
        "source": "MedlinePlus",
        "source_type": "health_topic",
        "url": "https://medlineplus.gov/edema.html",
        "section": "full_summary",
        "chunk_index": 0,
        "language": "English",
    },
    "distance": 0.25,
}

HEART_FAILURE_HIT = {
    "id": "medlineplus:health_topic:2:chunk:001",
    "document": "Heart failure can cause swelling in the legs.",
    "metadata": {
        "parent_id": "medlineplus:health_topic:2",
        "title": "Heart Failure",
        "source": "MedlinePlus",
        "source_type": "health_topic",
        "url": "https://medlineplus.gov/heartfailure.html",
        "section": "Symptoms",
        "chunk_index": 1,
    },
    "distance": 0.4,
}


class RetrieverTests(unittest.TestCase):
    def make_retriever(
        self,
        hits: list[dict[str, Any]] | None = None,
        embedder: FakeQueryEmbedder | None = None,
    ) -> tuple[Retriever, QueryableCollection, FakeQueryEmbedder]:
        collection = QueryableCollection(
            hits if hits is not None else [EDEMA_HIT, HEART_FAILURE_HIT]
        )
        index = ChromaCloudIndex(
            collection_name="test_collection",
            embedding_config=EmbeddingConfig(dimensions=3),
            client=FakeChromaClient(collection),
        )
        embedder = embedder or FakeQueryEmbedder()
        return Retriever(index, embedder), collection, embedder

    def test_returns_text_and_provenance(self) -> None:
        retriever, _, embedder = self.make_retriever()

        results = retriever.retrieve("bilateral lower-extremity edema", k=2)

        self.assertEqual(embedder.queries, ["bilateral lower-extremity edema"])
        self.assertEqual(len(results), 2)
        self.assertEqual(
            results[0],
            RetrievalResult(
                id="medlineplus:health_topic:1:chunk:000",
                parent_id="medlineplus:health_topic:1",
                title="Edema",
                text=(
                    "Edema is swelling caused by fluid in your "
                    "body's tissues."
                ),
                source="MedlinePlus",
                source_type="health_topic",
                url="https://medlineplus.gov/edema.html",
                section="full_summary",
                chunk_index=0,
                distance=0.25,
                similarity=0.75,
            ),
        )
        self.assertEqual(results[1].section, "Symptoms")
        self.assertAlmostEqual(results[1].similarity, 0.6)

    def test_passes_k_and_requests_provenance(self) -> None:
        retriever, collection, _ = self.make_retriever()

        results = retriever.retrieve("edema", k=1)

        self.assertEqual(len(results), 1)
        call = collection.query_calls[0]
        self.assertEqual(call["n_results"], 1)
        self.assertIsNone(call["where"])
        self.assertEqual(
            call["include"],
            ["documents", "metadatas", "distances"],
        )

    def test_filters_by_source_type(self) -> None:
        retriever, collection, _ = self.make_retriever()

        retriever.retrieve("vitamin D", k=5, source_type="glossary_definition")

        self.assertEqual(
            collection.query_calls[0]["where"],
            {"source_type": "glossary_definition"},
        )

    def test_empty_collection_returns_no_results(self) -> None:
        retriever, _, _ = self.make_retriever(hits=[])

        self.assertEqual(retriever.retrieve("edema"), [])

    def test_rejects_empty_query(self) -> None:
        retriever, collection, embedder = self.make_retriever()

        with self.assertRaises(ValueError):
            retriever.retrieve("   ")

        self.assertEqual(embedder.queries, [])
        self.assertEqual(collection.query_calls, [])

    def test_rejects_non_positive_k(self) -> None:
        retriever, _, _ = self.make_retriever()

        with self.assertRaises(ValueError):
            retriever.retrieve("edema", k=0)

    def test_rejects_wrong_query_vector_dimension(self) -> None:
        retriever, collection, _ = self.make_retriever(
            embedder=FakeQueryEmbedder(dimensions=4)
        )

        with self.assertRaises(ValueError):
            retriever.retrieve("edema")

        self.assertEqual(collection.query_calls, [])

    def test_result_serializes_to_dict(self) -> None:
        retriever, _, _ = self.make_retriever()

        result = retriever.retrieve("edema", k=1)[0].to_dict()

        self.assertEqual(result["title"], "Edema")
        self.assertEqual(result["url"], "https://medlineplus.gov/edema.html")
        self.assertEqual(result["similarity"], 0.75)


if __name__ == "__main__":
    unittest.main()
