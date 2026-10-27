from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import warnings
from typing import Any, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from retrieval.embeddings import EmbeddingConfig
from retrieval.ids import point_id
from retrieval.index import QdrantIndex
from retrieval.schema import RetrievalChunk
from scripts.build_retrieval_index import (
    batched,
    build_index,
    load_chunks,
    read_corpus_text,
)


def make_chunk(**overrides: object) -> RetrievalChunk:
    values: dict[str, object] = {
        "id": "chunk-1",
        "parent_id": "parent-1",
        "title": "Hypertension",
        "text": "High blood pressure can make the heart work harder.",
        "source": "MedlinePlus",
        "source_type": "health_topic",
        "url": "https://example.test/hypertension",
        "section": "full_summary",
        "chunk_index": 0,
        "synonyms": ["High blood pressure"],
    }
    values.update(overrides)
    return RetrievalChunk(**values)  # type: ignore[arg-type]


def make_index(
    *,
    client: QdrantClient | None = None,
    model: str = "gemini-embedding-2",
    dimensions: int = 3,
) -> QdrantIndex:
    """A real Qdrant index running in memory, with no server or network."""

    with warnings.catch_warnings():
        # In-memory Qdrant ignores payload indexes; the server honors them.
        warnings.filterwarnings(
            "ignore",
            message="Payload indexes have no effect in the local Qdrant",
        )
        return QdrantIndex(
            collection_name="test_collection",
            embedding_config=EmbeddingConfig(
                model=model,
                dimensions=dimensions,
            ),
            client=client or QdrantClient(":memory:"),
        )


class FakeEmbeddingClient:
    def __init__(self, dimensions: int = 3) -> None:
        self.dimensions = dimensions
        self.batch_sizes: list[int] = []

    def embed_documents(
        self,
        chunks: Sequence[RetrievalChunk],
    ) -> list[list[float]]:
        self.batch_sizes.append(len(chunks))
        return [
            [float(index + 1)] * self.dimensions
            for index, _ in enumerate(chunks)
        ]


class QdrantIndexTests(unittest.TestCase):
    def stored_point(
        self,
        index: QdrantIndex,
        chunk_id: str,
    ) -> Any:
        [record] = index.client.retrieve(
            collection_name=index.collection_name,
            ids=[point_id(chunk_id)],
            with_payload=True,
            with_vectors=True,
        )
        return record

    def test_loads_qdrant_settings_from_env_file(self) -> None:
        client_calls: list[dict[str, Any]] = []

        def qdrant_client(**kwargs: Any) -> QdrantClient:
            client_calls.append(kwargs)
            return QdrantClient(":memory:")

        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "QDRANT_URL=https://example.qdrant.test:6333\n"
                "QDRANT_API_KEY=test-key\n",
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch("qdrant_client.QdrantClient", qdrant_client),
            ):
                QdrantIndex(
                    collection_name="test_collection",
                    embedding_config=EmbeddingConfig(dimensions=3),
                    env_path=env_path,
                )

        self.assertEqual(client_calls, [{
            "url": "https://example.qdrant.test:6333",
            "api_key": "test-key",
        }])

    def test_requires_qdrant_url(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "QDRANT_URL"),
        ):
            QdrantIndex(env_path=None)

    def test_creates_cosine_collection_with_embedding_metadata(
        self,
    ) -> None:
        index = make_index()

        config = index.client.get_collection("test_collection").config
        self.assertEqual(config.params.vectors.size, 3)
        self.assertEqual(config.params.vectors.distance, Distance.COSINE)
        self.assertEqual(
            config.metadata,
            {
                "embedding_model": "gemini-embedding-2",
                "embedding_dimensions": 3,
                "distance_metric": "cosine",
            },
        )

    def test_reopens_an_existing_collection(self) -> None:
        client = QdrantClient(":memory:")
        make_index(client=client).upsert([make_chunk()], [[1.0, 2.0, 3.0]])

        self.assertEqual(make_index(client=client).count(), 1)

    def test_upsert_is_idempotent_by_stable_id(self) -> None:
        index = make_index()
        chunk = make_chunk()

        index.upsert([chunk], [[1.0, 2.0, 3.0]])
        index.upsert([chunk], [[3.0, 2.0, 1.0]])

        self.assertEqual(index.count(), 1)
        stored = self.stored_point(index, chunk.id)
        # Qdrant normalizes cosine vectors, so compare direction only.
        self.assertGreater(stored.vector[0], stored.vector[2])

    def test_upserts_original_text_and_provenance(self) -> None:
        index = make_index()
        chunk = make_chunk()

        index.upsert([chunk], [[1.0, 2.0, 3.0]])

        payload = self.stored_point(index, chunk.id).payload
        self.assertEqual(payload["chunk_id"], chunk.id)
        self.assertEqual(payload["text"], chunk.text)
        self.assertEqual(payload["source"], "MedlinePlus")
        self.assertEqual(payload["synonyms"], ["High blood pressure"])

    def test_returns_existing_ids(self) -> None:
        index = make_index()
        chunk = make_chunk()
        index.upsert([chunk], [[1.0, 2.0, 3.0]])

        self.assertEqual(
            index.existing_ids([chunk.id, "missing"]),
            {chunk.id},
        )

    def test_query_returns_closest_first_with_payload(self) -> None:
        index = make_index()
        index.upsert(
            [
                make_chunk(id="near", title="Edema"),
                make_chunk(id="far", title="Asthma"),
            ],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        hits = index.query([1.0, 0.1, 0.0], k=2)

        self.assertEqual([hit["id"] for hit in hits], ["near", "far"])
        self.assertEqual(hits[0]["metadata"]["title"], "Edema")
        self.assertEqual(hits[0]["document"], make_chunk().text)
        self.assertNotIn("chunk_id", hits[0]["metadata"])
        self.assertGreater(hits[0]["score"], hits[1]["score"])

    def test_query_filters_on_payload_fields(self) -> None:
        index = make_index()
        index.upsert(
            [
                make_chunk(id="topic", source_type="health_topic"),
                make_chunk(id="term", source_type="glossary_definition"),
            ],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        hits = index.query(
            [1.0, 0.0, 0.0],
            k=5,
            where={"source_type": "glossary_definition"},
        )

        self.assertEqual([hit["id"] for hit in hits], ["term"])

    def test_rejects_wrong_embedding_dimension(self) -> None:
        index = make_index()

        with self.assertRaisesRegex(ValueError, "expected 3"):
            index.upsert([make_chunk()], [[1.0, 2.0]])

        with self.assertRaisesRegex(ValueError, "expected 3"):
            index.query([1.0, 2.0], k=1)

    def test_rejects_existing_non_cosine_collection(self) -> None:
        client = QdrantClient(":memory:")
        client.create_collection(
            "test_collection",
            vectors_config=VectorParams(size=3, distance=Distance.EUCLID),
            metadata={"embedding_model": "gemini-embedding-2"},
        )

        with self.assertRaisesRegex(ValueError, "expected 'cosine'"):
            make_index(client=client)

    def test_rejects_existing_dimension_mismatch(self) -> None:
        client = QdrantClient(":memory:")
        make_index(client=client, dimensions=4)

        with self.assertRaisesRegex(ValueError, "embedding_dimensions"):
            make_index(client=client, dimensions=3)

    def test_rejects_existing_model_mismatch(self) -> None:
        client = QdrantClient(":memory:")
        make_index(client=client, model="other-model")

        with self.assertRaisesRegex(ValueError, "embedding_model"):
            make_index(client=client)


class BuildIndexTests(unittest.TestCase):
    def test_retries_timed_out_corpus_read(self) -> None:
        path = Path("cloud-backed-corpus.jsonl")

        with (
            patch.object(
                Path,
                "read_text",
                side_effect=[TimeoutError(), "ready"],
            ) as read_text,
            patch("scripts.build_retrieval_index.time.sleep") as sleep,
        ):
            text = read_corpus_text(
                path,
                attempts=2,
                retry_seconds=0.01,
            )

        self.assertEqual(text, "ready")
        self.assertEqual(read_text.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_explains_how_to_materialize_timed_out_corpus(self) -> None:
        path = Path("cloud-backed-corpus.jsonl")

        with patch.object(
            Path,
            "read_text",
            side_effect=TimeoutError(),
        ):
            with self.assertRaisesRegex(RuntimeError, "Keep Downloaded"):
                read_corpus_text(
                    path,
                    attempts=1,
                    retry_seconds=0,
                )

    def test_loads_jsonl_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "chunks.jsonl"
            path.write_text(
                json.dumps(make_chunk().to_dict()) + "\n",
                encoding="utf-8",
            )

            chunks = load_chunks(path)

        self.assertEqual(chunks, [make_chunk()])

    def test_batches_values(self) -> None:
        self.assertEqual(
            [list(batch) for batch in batched([1, 2, 3, 4, 5], 2)],
            [[1, 2], [3, 4], [5]],
        )

    def test_build_index_embeds_and_upserts_in_batches(self) -> None:
        index = make_index()
        embedding_client = FakeEmbeddingClient()
        chunks = [
            make_chunk(id=f"chunk-{index}", chunk_index=index)
            for index in range(5)
        ]

        final_count = build_index(
            chunks,
            index=index,
            embedding_client=embedding_client,  # type: ignore[arg-type]
            batch_size=2,
        )

        self.assertEqual(embedding_client.batch_sizes, [2, 2, 1])
        self.assertEqual(final_count, 5)

    def test_build_index_skips_records_from_an_interrupted_run(self) -> None:
        index = make_index()
        existing = make_chunk(id="chunk-0", chunk_index=0)
        index.upsert([existing], [[1.0, 1.0, 1.0]])
        embedding_client = FakeEmbeddingClient()
        chunks = [
            existing,
            make_chunk(id="chunk-1", chunk_index=1),
        ]

        final_count = build_index(
            chunks,
            index=index,
            embedding_client=embedding_client,  # type: ignore[arg-type]
            batch_size=2,
        )

        self.assertEqual(final_count, 2)
        self.assertEqual(embedding_client.batch_sizes, [1])


if __name__ == "__main__":
    unittest.main()
