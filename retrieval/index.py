from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from dotenv import load_dotenv

from retrieval.embeddings import (
    DEFAULT_EMBEDDING_CONFIG,
    EmbeddingConfig,
)
from retrieval.ids import point_id
from retrieval.schema import RetrievalChunk

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


DEFAULT_COLLECTION_NAME = "faithfulmed_lay_health_gemini2_768_v1"
DEFAULT_ENV_PATH = Path(".env")
DISTANCE_METRIC = "cosine"

# Payload keys added next to RetrievalChunk.to_payload() fields.
CHUNK_ID_KEY = "chunk_id"
TEXT_KEY = "text"

# Payload fields used in retrieval filters get a keyword index.
FILTERABLE_FIELDS = ("source_type", "source")


class QdrantIndex:
    """Shared Qdrant collection backed by Gemini embeddings."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
        *,
        env_path: Path | str | None = DEFAULT_ENV_PATH,
        client: QdrantClient | None = None,
    ) -> None:
        if not collection_name.strip():
            raise ValueError("Qdrant collection name must not be empty.")

        self.collection_name = collection_name
        self.embedding_config = embedding_config

        if client is None:
            if env_path is not None:
                load_dotenv(dotenv_path=env_path)

            url = os.environ.get("QDRANT_URL")

            if not url:
                raise ValueError(
                    "Missing Qdrant environment variable: QDRANT_URL"
                )

            from qdrant_client import QdrantClient

            # QDRANT_API_KEY is optional so a local Docker Qdrant works too.
            client = QdrantClient(
                url=url,
                api_key=os.environ.get("QDRANT_API_KEY") or None,
            )

        self._client = client

        if not client.collection_exists(collection_name):
            self._create_collection()

        self._validate_collection_contract()

    @property
    def client(self) -> QdrantClient:
        return self._client

    def _create_collection(self) -> None:
        from qdrant_client import models

        self._client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_config.dimensions,
                distance=models.Distance.COSINE,
            ),
            metadata={
                "embedding_model": self.embedding_config.model,
                "embedding_dimensions": self.embedding_config.dimensions,
                "distance_metric": DISTANCE_METRIC,
            },
        )

        for field_name in FILTERABLE_FIELDS:
            self._client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _validate_collection_contract(self) -> None:
        config = self._client.get_collection(self.collection_name).config
        vectors = config.params.vectors

        if isinstance(vectors, dict):
            raise ValueError(
                f"Collection {self.collection_name!r} uses named vectors; "
                "expected a single unnamed vector."
            )

        actual_metric = str(vectors.distance.value).lower()

        if actual_metric != DISTANCE_METRIC:
            raise ValueError(
                f"Collection {self.collection_name!r} uses "
                f"{actual_metric!r} distance; expected {DISTANCE_METRIC!r}."
            )

        if vectors.size != self.embedding_config.dimensions:
            raise ValueError(
                f"Collection {self.collection_name!r} has "
                f"embedding_dimensions={vectors.size!r}; expected "
                f"{self.embedding_config.dimensions!r}."
            )

        metadata = config.metadata or {}
        actual_model = metadata.get("embedding_model")

        if actual_model != self.embedding_config.model:
            raise ValueError(
                f"Collection {self.collection_name!r} has "
                f"embedding_model={actual_model!r}; expected "
                f"{self.embedding_config.model!r}."
            )

    def upsert(
        self,
        chunks: Sequence[RetrievalChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        """Create or replace chunk records by their stable IDs."""

        from qdrant_client import models

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Received {len(chunks)} chunks and "
                f"{len(embeddings)} embeddings."
            )

        if not chunks:
            return 0

        ids = [chunk.id for chunk in chunks]

        if len(ids) != len(set(ids)):
            raise ValueError("A Qdrant upsert batch contains duplicate IDs.")

        vectors = [
            self._validated_vector(embedding, index)
            for index, embedding in enumerate(embeddings)
        ]

        self._client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id(chunk.id),
                    vector=vector,
                    payload={
                        **chunk.to_payload(),
                        CHUNK_ID_KEY: chunk.id,
                        TEXT_KEY: chunk.text,
                    },
                )
                for chunk, vector in zip(chunks, vectors)
            ],
            wait=True,
        )

        return len(chunks)

    def query(
        self,
        embedding: Sequence[float],
        *,
        k: int,
        where: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Return the k nearest records, most similar first.

        where is an exact-match filter on payload fields, e.g.
        {"source_type": "health_topic"}.
        """

        from qdrant_client import models

        if k <= 0:
            raise ValueError("k must be positive.")

        vector = self._validated_vector(embedding, 0)

        query_filter = None

        if where:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                    for key, value in where.items()
                ]
            )

        response = self._client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
        )

        hits: list[dict[str, Any]] = []

        for point in response.points:
            payload = dict(point.payload or {})
            hits.append({
                "id": payload.pop(CHUNK_ID_KEY, str(point.id)),
                "document": payload.pop(TEXT_KEY, ""),
                "metadata": payload,
                # Qdrant reports cosine similarity (higher is closer).
                "score": point.score,
            })

        return hits

    def count(self) -> int:
        return self._client.count(
            collection_name=self.collection_name,
            exact=True,
        ).count

    def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Return the requested chunk IDs that are already stored."""

        if not ids:
            return set()

        chunk_ids_by_point = {point_id(chunk_id): chunk_id for chunk_id in ids}

        records = self._client.retrieve(
            collection_name=self.collection_name,
            ids=list(chunk_ids_by_point),
            with_payload=False,
            with_vectors=False,
        )

        return {chunk_ids_by_point[str(record.id)] for record in records}

    def _validated_vector(
        self,
        embedding: Sequence[float],
        index: int,
    ) -> list[float]:
        vector = [float(value) for value in embedding]

        if len(vector) != self.embedding_config.dimensions:
            raise ValueError(
                f"Embedding {index} has {len(vector)} dimensions; "
                f"expected {self.embedding_config.dimensions}."
            )

        return vector
