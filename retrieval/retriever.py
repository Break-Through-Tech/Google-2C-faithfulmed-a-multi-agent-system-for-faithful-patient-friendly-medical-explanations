from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Protocol

from dotenv import load_dotenv

from retrieval.embeddings import GeminiEmbeddingClient
from retrieval.index import DEFAULT_ENV_PATH, ChromaCloudIndex


DEFAULT_K = 5


@dataclass(slots=True)
class RetrievalResult:
    """One retrieved chunk with the provenance needed to cite it."""

    id: str
    parent_id: str

    title: str
    text: str

    source: str
    source_type: str
    url: str

    section: str
    chunk_index: int

    # Cosine distance from Chroma; similarity = 1 - distance.
    distance: float
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_hit(cls, hit: dict[str, Any]) -> RetrievalResult:
        """Build a result from one ChromaCloudIndex.query() record."""

        metadata = hit["metadata"]
        distance = float(hit["distance"])

        return cls(
            id=hit["id"],
            parent_id=metadata.get("parent_id", ""),
            title=metadata.get("title", ""),
            text=hit["document"] or "",
            source=metadata.get("source", ""),
            source_type=metadata.get("source_type", ""),
            url=metadata.get("url", ""),
            section=metadata.get("section", ""),
            chunk_index=int(metadata.get("chunk_index", 0)),
            distance=distance,
            similarity=1.0 - distance,
        )


class _QueryEmbedder(Protocol):
    def embed_query(self, query: str) -> list[float]: ...


class Retriever:
    """Embed a query with Gemini and search the shared Chroma collection."""

    def __init__(
        self,
        index: ChromaCloudIndex,
        embedding_client: _QueryEmbedder,
    ) -> None:
        self.index = index
        self.embedding_client = embedding_client

    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_K,
        *,
        source_type: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Return the k most similar chunks, most similar first.

        source_type optionally restricts results, e.g. "health_topic"
        or "glossary_definition".
        """

        if not query.strip():
            raise ValueError("Retrieval query must not be empty.")

        if k <= 0:
            raise ValueError("k must be positive.")

        where = {"source_type": source_type} if source_type else None
        embedding = self.embedding_client.embed_query(query)
        hits = self.index.query(embedding, k=k, where=where)

        return [RetrievalResult.from_hit(hit) for hit in hits]


@lru_cache(maxsize=1)
def get_default_retriever() -> Retriever:
    """Build (once) a retriever from the shared .env configuration."""

    # The Gemini SDK reads GEMINI_API_KEY from the environment, so .env
    # must be loaded before the embedding client is created.
    load_dotenv(dotenv_path=DEFAULT_ENV_PATH)

    index = ChromaCloudIndex()

    return Retriever(
        index=index,
        embedding_client=GeminiEmbeddingClient(
            config=index.embedding_config
        ),
    )


def retrieve(
    query: str,
    k: int = DEFAULT_K,
    *,
    source_type: str | None = None,
) -> list[RetrievalResult]:
    """
    Search the FaithfulMed lay-health index.

        results = retrieve("bilateral lower-extremity edema", k=5)
    """

    return get_default_retriever().retrieve(
        query,
        k,
        source_type=source_type,
    )
