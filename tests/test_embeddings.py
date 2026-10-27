from __future__ import annotations

from dataclasses import dataclass
import unittest
from unittest.mock import patch
from typing import Any

from retrieval.embeddings import (
    EmbeddingConfig,
    GeminiEmbeddingClient,
    build_document_embedding_text,
    build_query_embedding_text,
)
from retrieval.schema import RetrievalChunk


def make_chunk(**overrides: object) -> RetrievalChunk:
    values: dict[str, object] = {
        "id": "chunk-1",
        "parent_id": "parent-1",
        "title": "High Blood Pressure",
        "text": "High blood pressure can make your heart work harder.",
        "source": "MedlinePlus",
        "source_type": "health_topic",
        "url": "https://example.test/high-blood-pressure",
        "section": "Treatment",
        "chunk_index": 0,
        "synonyms": ["Hypertension", "High blood pressure"],
        "see_references": ["Hypertension", "Elevated blood pressure"],
        "mesh_terms": ["Hypertension", "Hypertension"],
    }
    values.update(overrides)
    return RetrievalChunk(**values)  # type: ignore[arg-type]


@dataclass
class FakeEmbedding:
    values: list[float] | None


@dataclass
class FakeResponse:
    embeddings: list[FakeEmbedding] | None


class FakeModels:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions
        self.calls: list[dict[str, Any]] = []

    def embed_content(self, **kwargs: Any) -> FakeResponse:
        self.calls.append(kwargs)
        embeddings = [
            FakeEmbedding([float(index)] * self.dimensions)
            for index, _ in enumerate(kwargs["contents"])
        ]
        return FakeResponse(embeddings)


class FakeClient:
    def __init__(self, dimensions: int) -> None:
        self.models = FakeModels(dimensions)


class RateLimitError(Exception):
    code = 429


class EmbeddingFormatterTests(unittest.TestCase):
    def test_document_formatter_uses_only_semantic_fields(self) -> None:
        formatted = build_document_embedding_text(make_chunk())

        self.assertEqual(
            formatted,
            "title: High Blood Pressure | text: "
            "Also called: Hypertension, High blood pressure, "
            "Elevated blood pressure\n"
            "Medical terms: Hypertension\n"
            "Section: Treatment\n"
            "High blood pressure can make your heart work harder.",
        )
        self.assertNotIn("MedlinePlus", formatted)
        self.assertNotIn("chunk-1", formatted)

    def test_document_formatter_omits_placeholder_sections(self) -> None:
        for section in ("definition", "full_summary"):
            with self.subTest(section=section):
                formatted = build_document_embedding_text(
                    make_chunk(section=section)
                )
                self.assertNotIn("Section:", formatted)

    def test_document_formatter_rejects_empty_text(self) -> None:
        with self.assertRaises(ValueError):
            build_document_embedding_text(make_chunk(text="  "))

    def test_query_formatter_adds_retrieval_instruction(self) -> None:
        self.assertEqual(
            build_query_embedding_text("  What is hypertension?  "),
            "task: search result | query: What is hypertension?",
        )

    def test_query_formatter_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            build_query_embedding_text("  ")


class GeminiEmbeddingClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = EmbeddingConfig(
            model="gemini-embedding-2",
            dimensions=3,
        )
        self.fake_client = FakeClient(dimensions=3)
        self.client = GeminiEmbeddingClient(
            config=self.config,
            client=self.fake_client,
        )

    def test_embeds_documents_as_separate_contents(self) -> None:
        vectors = self.client.embed_documents([
            make_chunk(),
            make_chunk(id="chunk-2", title="Diabetes"),
        ])

        self.assertEqual(vectors, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

        [call] = self.fake_client.models.calls
        self.assertEqual(call["model"], "gemini-embedding-2")
        self.assertEqual(call["config"], {"output_dimensionality": 3})
        self.assertEqual(len(call["contents"]), 2)
        self.assertIn(
            "title: High Blood Pressure",
            call["contents"][0]["parts"][0]["text"],
        )

    def test_embeds_query_with_the_same_model_and_dimensions(self) -> None:
        vector = self.client.embed_query("hypertension")

        self.assertEqual(vector, [0.0, 0.0, 0.0])
        [call] = self.fake_client.models.calls
        self.assertEqual(
            call["contents"],
            [{
                "parts": [{
                    "text": "task: search result | query: hypertension"
                }]
            }],
        )

    def test_empty_document_batch_makes_no_request(self) -> None:
        self.assertEqual(self.client.embed_documents([]), [])
        self.assertEqual(self.fake_client.models.calls, [])

    def test_rejects_wrong_vector_dimensions(self) -> None:
        client = GeminiEmbeddingClient(
            config=self.config,
            client=FakeClient(dimensions=2),
        )

        with self.assertRaisesRegex(RuntimeError, "expected 3"):
            client.embed_query("hypertension")

    def test_rejects_response_count_mismatch(self) -> None:
        class MissingModels:
            def embed_content(self, **kwargs: Any) -> FakeResponse:
                return FakeResponse([])

        class MissingClient:
            models = MissingModels()

        client = GeminiEmbeddingClient(
            config=self.config,
            client=MissingClient(),
        )

        with self.assertRaisesRegex(RuntimeError, "0 embeddings for 1 inputs"):
            client.embed_query("hypertension")

    def test_retries_rate_limit_using_server_delay(self) -> None:
        class RateLimitedModels(FakeModels):
            def embed_content(self, **kwargs: Any) -> FakeResponse:
                if not self.calls:
                    self.calls.append(kwargs)
                    raise RateLimitError("Please retry in 7.5s.")

                return super().embed_content(**kwargs)

        class RateLimitedClient:
            models = RateLimitedModels(dimensions=3)

        client = GeminiEmbeddingClient(
            config=self.config,
            client=RateLimitedClient(),
        )

        with patch("retrieval.embeddings.time.sleep") as sleep:
            vector = client.embed_query("hypertension")

        self.assertEqual(vector, [0.0, 0.0, 0.0])
        sleep.assert_called_once_with(10.0)

    def test_does_not_retry_non_rate_limit_errors(self) -> None:
        class FailingModels:
            def embed_content(self, **kwargs: Any) -> FakeResponse:
                raise ValueError("bad request")

        class FailingClient:
            models = FailingModels()

        client = GeminiEmbeddingClient(
            config=self.config,
            client=FailingClient(),
        )

        with (
            patch("retrieval.embeddings.time.sleep") as sleep,
            self.assertRaisesRegex(ValueError, "bad request"),
        ):
            client.embed_query("hypertension")

        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
