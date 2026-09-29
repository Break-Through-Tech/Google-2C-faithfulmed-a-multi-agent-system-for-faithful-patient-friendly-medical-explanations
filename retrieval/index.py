from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol, Sequence

from dotenv import load_dotenv

from retrieval.embeddings import (
    DEFAULT_EMBEDDING_CONFIG,
    EmbeddingConfig,
)
from retrieval.schema import RetrievalChunk


DEFAULT_COLLECTION_NAME = "faithfulmed_lay_health_gemini2_768_v1"
DEFAULT_ENV_PATH = Path(".env")
DISTANCE_METRIC = "cosine"
VECTOR_INDEX_TYPE = "spann"


class _Collection(Protocol):
    metadata: dict[str, Any] | None
    configuration_json: dict[str, Any]

    def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None: ...

    def count(self) -> int: ...

    def get(
        self,
        *,
        ids: list[str],
        include: list[str],
    ) -> dict[str, Any]: ...

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None,
        include: list[str],
    ) -> dict[str, Any]: ...


class _ChromaClient(Protocol):
    def get_or_create_collection(
        self,
        *,
        name: str,
        configuration: dict[str, Any],
        metadata: dict[str, Any],
        embedding_function: None,
    ) -> _Collection: ...


class ChromaCloudIndex:
    """Shared Chroma Cloud collection backed by Gemini embeddings."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
        *,
        env_path: Path | str | None = DEFAULT_ENV_PATH,
        client: _ChromaClient | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("Chroma collection name must not be empty.")

        self.collection_name = collection_name
        self.embedding_config = embedding_config

        if client is None:
            if env_path is not None:
                load_dotenv(dotenv_path=env_path)

            required_variables = (
                "CHROMA_API_KEY",
                "CHROMA_TENANT",
                "CHROMA_DATABASE",
            )
            missing_variables = [
                name
                for name in required_variables
                if not os.environ.get(name)
            ]

            if missing_variables:
                raise ValueError(
                    "Missing Chroma Cloud environment variables: "
                    + ", ".join(missing_variables)
                )

            import chromadb

            client = chromadb.CloudClient(
                api_key=os.environ["CHROMA_API_KEY"],
                tenant=os.environ["CHROMA_TENANT"],
                database=os.environ["CHROMA_DATABASE"],
            )

        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection_name,
            configuration={
                VECTOR_INDEX_TYPE: {
                    "space": DISTANCE_METRIC,
                }
            },
            metadata={
                "embedding_model": embedding_config.model,
                "embedding_dimensions": embedding_config.dimensions,
                "distance_metric": DISTANCE_METRIC,
            },
            # Embeddings are produced by Gemini, never by Chroma's default
            # local embedding function.
            embedding_function=None,
        )

        self._validate_collection_contract()

    @property
    def collection(self) -> _Collection:
        return self._collection

    def _validate_collection_contract(self) -> None:
        configuration = self._collection.configuration_json
        vector_config = (
            configuration.get("spann")
            or configuration.get("hnsw")
            or {}
        )
        actual_metric = vector_config.get("space")

        if actual_metric != DISTANCE_METRIC:
            raise ValueError(
                f"Collection {self.collection_name!r} uses "
                f"{actual_metric!r} distance; expected {DISTANCE_METRIC!r}."
            )

        metadata = self._collection.metadata or {}
        expected_metadata = {
            "embedding_model": self.embedding_config.model,
            "embedding_dimensions": self.embedding_config.dimensions,
            "distance_metric": DISTANCE_METRIC,
        }

        for key, expected_value in expected_metadata.items():
            actual_value = metadata.get(key)

            if actual_value != expected_value:
                raise ValueError(
                    f"Collection {self.collection_name!r} has {key}="
                    f"{actual_value!r}; expected {expected_value!r}."
                )

    def upsert(
        self,
        chunks: Sequence[RetrievalChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Create or replace chunk records by their stable IDs."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Received {len(chunks)} chunks and "
                f"{len(embeddings)} embeddings."
            )

        if not chunks:
            return 0

        ids = [chunk.id for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("A Chroma upsert batch contains duplicate IDs.")

        vectors = [
            self._validated_vector(embedding, index)
            for index, embedding in enumerate(embeddings)
        ]

        self._collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.to_chroma_metadata() for chunk in chunks],
        )

        return len(chunks)

    def query(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the k nearest records, closest first."""

        if k <= 0:
            raise ValueError("k must be positive.")

        vector = self._validated_vector(embedding, 0)

        result = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns one nested list per query embedding.
        ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            {
                "id": ids[position],
                "document": documents[position],
                "metadata": metadatas[position] or {},
                "distance": distances[position],
            }
            for position in range(len(ids))
        ]

    def count(self) -> int:
        return self._collection.count()

    def _validated_vector(
        self,
        embedding: Sequence[float],
        index: int,
    ) -> list[float]:
        vector = list(embedding)

        if len(vector) != self.embedding_config.dimensions:
            raise ValueError(
                f"Embedding {index} has {len(vector)} dimensions; "
                f"expected {self.embedding_config.dimensions}."
            )

        return vector

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Return the requested IDs that are already stored in Chroma."""

        if not ids:
            return set()

        result = self._collection.get(
            ids=list(ids),
            include=[],
        )
        return set(result["ids"])
