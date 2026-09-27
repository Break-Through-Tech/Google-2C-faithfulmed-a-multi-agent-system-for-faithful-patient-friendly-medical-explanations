from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from typing import Any, Sequence

from retrieval.embeddings import EmbeddingConfig
from retrieval.index import ChromaCloudIndex
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


class FakeCollection:
    def __init__(
        self,
        *,
        model: str = "gemini-embedding-2",
        dimensions: int = 3,
        metric: str = "cosine",
        index_type: str = "spann",
    ) -> None:
        self.metadata = {
            "embedding_model": model,
            "embedding_dimensions": dimensions,
            "distance_metric": metric,
        }
        self.configuration_json = {
            "hnsw": None,
            "spann": None,
        }
        self.configuration_json[index_type] = {"space": metric}
        self.records: dict[str, dict[str, Any]] = {}
        self.upsert_calls = 0

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_calls += 1

        for index, chunk_id in enumerate(kwargs["ids"]):
            self.records[chunk_id] = {
                "embedding": kwargs["embeddings"][index],
                "document": kwargs["documents"][index],
                "metadata": kwargs["metadatas"][index],
            }

    def count(self) -> int:
        return len(self.records)

    def get(self, **kwargs: Any) -> dict[str, list[str]]:
        return {
            "ids": [
                chunk_id
                for chunk_id in kwargs["ids"]
                if chunk_id in self.records
            ]
        }


class FakeChromaClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.calls: list[dict[str, Any]] = []

    def get_or_create_collection(self, **kwargs: Any) -> FakeCollection:
        self.calls.append(kwargs)
        return self.collection


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
            [float(index)] * self.dimensions
            for index, _ in enumerate(chunks)
        ]


class ChromaCloudIndexTests(unittest.TestCase):
    def make_index(
        self,
        collection: FakeCollection | None = None,
    ) -> tuple[ChromaCloudIndex, FakeChromaClient]:
        collection = collection or FakeCollection()
        client = FakeChromaClient(collection)
        index = ChromaCloudIndex(
            collection_name="test_collection",
            embedding_config=EmbeddingConfig(dimensions=3),
            client=client,
        )
        return index, client

    def test_loads_cloud_credentials_from_env_file(self) -> None:
        collection = FakeCollection()
        cloud_calls: list[dict[str, str]] = []

        def cloud_client(**kwargs: str) -> FakeChromaClient:
            cloud_calls.append(kwargs)
            return FakeChromaClient(collection)

        fake_chromadb = SimpleNamespace(CloudClient=cloud_client)

        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "CHROMA_API_KEY=test-key\n"
                "CHROMA_TENANT=test-tenant\n"
                "CHROMA_DATABASE=test-database\n",
                encoding="utf-8",
            )

            with (
                patch.dict(os.environ, {}, clear=True),
                patch.dict(sys.modules, {"chromadb": fake_chromadb}),
            ):
                ChromaCloudIndex(
                    collection_name="test_collection",
                    embedding_config=EmbeddingConfig(dimensions=3),
                    env_path=env_path,
                )

        self.assertEqual(cloud_calls, [{
            "api_key": "test-key",
            "tenant": "test-tenant",
            "database": "test-database",
        }])

    def test_creates_cosine_spann_collection_without_default_embedder(
        self,
    ) -> None:
        _, client = self.make_index()

        [call] = client.calls
        self.assertEqual(
            call["configuration"],
            {"spann": {"space": "cosine"}},
        )
        self.assertIsNone(call["embedding_function"])
        self.assertEqual(call["metadata"]["embedding_dimensions"], 3)

    def test_upsert_is_idempotent_by_stable_id(self) -> None:
        index, client = self.make_index()
        chunk = make_chunk()

        index.upsert([chunk], [[1.0, 2.0, 3.0]])
        index.upsert([chunk], [[3.0, 2.0, 1.0]])

        self.assertEqual(index.count(), 1)
        self.assertEqual(client.collection.upsert_calls, 2)
        self.assertEqual(
            client.collection.records[chunk.id]["embedding"],
            [3.0, 2.0, 1.0],
        )

    def test_upserts_original_text_and_provenance(self) -> None:
        index, client = self.make_index()
        chunk = make_chunk()

        index.upsert([chunk], [[1.0, 2.0, 3.0]])

        record = client.collection.records[chunk.id]
        self.assertEqual(record["document"], chunk.text)
        self.assertEqual(record["metadata"]["source"], "MedlinePlus")
        self.assertEqual(
            record["metadata"]["synonyms"],
            ["High blood pressure"],
        )

    def test_returns_existing_ids(self) -> None:
        index, _ = self.make_index()
        chunk = make_chunk()
        index.upsert([chunk], [[1.0, 2.0, 3.0]])

        self.assertEqual(
            index.existing_ids([chunk.id, "missing"]),
            {chunk.id},
        )

    def test_rejects_wrong_embedding_dimension(self) -> None:
        index, _ = self.make_index()

        with self.assertRaisesRegex(ValueError, "expected 3"):
            index.upsert([make_chunk()], [[1.0, 2.0]])

    def test_rejects_existing_non_cosine_collection(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 'cosine'"):
            self.make_index(FakeCollection(metric="l2"))

    def test_rejects_existing_model_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "embedding_model"):
            self.make_index(FakeCollection(model="other-model"))


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
        collection = FakeCollection()
        index, _ = ChromaCloudIndexTests().make_index(collection)
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
        self.assertEqual(collection.upsert_calls, 3)

    def test_build_index_skips_records_from_an_interrupted_run(self) -> None:
        collection = FakeCollection()
        index, _ = ChromaCloudIndexTests().make_index(collection)
        existing = make_chunk(id="chunk-0", chunk_index=0)
        index.upsert([existing], [[1.0, 1.0, 1.0]])
        collection.upsert_calls = 0
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
        self.assertEqual(collection.upsert_calls, 1)


if __name__ == "__main__":
    unittest.main()
