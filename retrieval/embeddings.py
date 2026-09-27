from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Protocol, Sequence

from retrieval.schema import RetrievalChunk


DEFAULT_EMBEDDING_MODEL = "gemini-embedding-2"
DEFAULT_EMBEDDING_DIMENSIONS = 768

_NON_SEMANTIC_SECTIONS = {
    "definition",
    "full_summary",
}

_RETRY_DELAY_RE = re.compile(
    r"retry in\s+([0-9]+(?:\.[0-9]+)?)s",
    re.IGNORECASE,
)
_MAX_RATE_LIMIT_RETRIES = 6
_INITIAL_RATE_LIMIT_DELAY_SECONDS = 10.0
_MAX_RATE_LIMIT_DELAY_SECONDS = 65.0


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Gemini model settings shared by document and query embeddings."""

    model: str = DEFAULT_EMBEDDING_MODEL
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("Embedding model must not be empty.")

        if self.dimensions <= 0:
            raise ValueError("Embedding dimensions must be positive.")


DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig()


def _unique_nonempty(values: Sequence[str]) -> list[str]:
    """Remove empty and duplicate vocabulary while preserving source order."""

    unique: list[str] = []
    seen: set[str] = set()

    for value in values:
        clean_value = value.strip()

        if clean_value and clean_value not in seen:
            unique.append(clean_value)
            seen.add(clean_value)

    return unique


def build_document_embedding_text(
    chunk: RetrievalChunk,
) -> str:
    """Format only semantically useful chunk fields for retrieval embedding."""

    context: list[str] = []
    aliases = _unique_nonempty(
        [*chunk.synonyms, *chunk.see_references]
    )

    if aliases:
        context.append("Also called: " + ", ".join(aliases))

    mesh_terms = _unique_nonempty(chunk.mesh_terms)

    if mesh_terms:
        context.append("Medical terms: " + ", ".join(mesh_terms))

    section = chunk.section.strip()

    if section and section not in _NON_SEMANTIC_SECTIONS:
        context.append(f"Section: {section}")

    text = chunk.text.strip()

    if not text:
        raise ValueError(f"Chunk {chunk.id!r} has no text to embed.")

    context.append(text)

    title = chunk.title.strip() or "none"
    content = "\n".join(context)

    return f"title: {title} | text: {content}"


def build_query_embedding_text(query: str) -> str:
    """Format a query using Gemini Embedding 2's retrieval instruction."""

    clean_query = query.strip()

    if not clean_query:
        raise ValueError("Embedding query must not be empty.")

    return f"task: search result | query: {clean_query}"


class _ModelsClient(Protocol):
    def embed_content(
        self,
        *,
        model: str,
        contents: list[dict[str, Any]],
        config: dict[str, int],
    ) -> Any: ...


class _GenAIClient(Protocol):
    models: _ModelsClient


class GeminiEmbeddingClient:
    """Small, validated adapter around Google Gen AI embedding requests."""

    def __init__(
        self,
        config: EmbeddingConfig = DEFAULT_EMBEDDING_CONFIG,
        *,
        api_key: str | None = None,
        client: _GenAIClient | None = None,
    ) -> None:
        if client is not None and api_key is not None:
            raise ValueError(
                "Pass either an injected client or api_key, not both."
            )

        self.config = config

        if client is None:
            # Keep SDK startup out of formatter-only workflows and tests.
            from google import genai

            client_kwargs = {}

            if api_key is not None:
                client_kwargs["api_key"] = api_key

            client = genai.Client(**client_kwargs)

        self._client = client

    def embed_document(self, chunk: RetrievalChunk) -> list[float]:
        """Embed one corpus chunk."""

        return self._embed_texts([
            build_document_embedding_text(chunk)
        ])[0]

    def embed_documents(
        self,
        chunks: Sequence[RetrievalChunk],
    ) -> list[list[float]]:
        """Embed corpus chunks in input order."""

        if not chunks:
            return []

        return self._embed_texts([
            build_document_embedding_text(chunk)
            for chunk in chunks
        ])

    def embed_query(self, query: str) -> list[float]:
        """Embed one search query in the document embedding space."""

        return self._embed_texts([
            build_query_embedding_text(query)
        ])[0]

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        # Gemini Embedding 2 aggregates parts in one Content. Wrapping every
        # string as its own Content requests one vector per input string.
        contents = [
            {"parts": [{"text": text}]}
            for text in texts
        ]

        response = self._request_with_rate_limit_retry(contents)

        response_embeddings = response.embeddings

        if response_embeddings is None:
            raise RuntimeError("Gemini returned no embeddings.")

        if len(response_embeddings) != len(texts):
            raise RuntimeError(
                "Gemini returned "
                f"{len(response_embeddings)} embeddings for "
                f"{len(texts)} inputs."
            )

        vectors: list[list[float]] = []

        for index, embedding in enumerate(response_embeddings):
            values = embedding.values

            if values is None:
                raise RuntimeError(
                    f"Gemini embedding {index} has no vector values."
                )

            vector = list(values)

            if len(vector) != self.config.dimensions:
                raise RuntimeError(
                    f"Gemini embedding {index} has {len(vector)} dimensions; "
                    f"expected {self.config.dimensions}."
                )

            vectors.append(vector)

        return vectors

    def _request_with_rate_limit_retry(
        self,
        contents: list[dict[str, Any]],
    ) -> Any:
        for retry_number in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                return self._client.models.embed_content(
                    model=self.config.model,
                    contents=contents,
                    config={
                        "output_dimensionality": self.config.dimensions,
                    },
                )
            except Exception as error:
                if (
                    getattr(error, "code", None) != 429
                    or retry_number == _MAX_RATE_LIMIT_RETRIES
                ):
                    raise

                delay = self._rate_limit_delay(
                    error,
                    retry_number,
                )
                print(
                    "Gemini rate limit reached; retrying in "
                    f"{delay:.1f}s "
                    f"({retry_number + 1}/{_MAX_RATE_LIMIT_RETRIES})..."
                )
                time.sleep(delay)

        raise AssertionError("Unreachable Gemini retry state.")

    @staticmethod
    def _rate_limit_delay(
        error: Exception,
        retry_number: int,
    ) -> float:
        fallback_delay = min(
            _INITIAL_RATE_LIMIT_DELAY_SECONDS * (2 ** retry_number),
            _MAX_RATE_LIMIT_DELAY_SECONDS,
        )
        match = _RETRY_DELAY_RE.search(str(error))

        if match is None:
            return fallback_delay

        server_delay = float(match.group(1)) + 1.0
        return min(
            max(server_delay, fallback_delay),
            _MAX_RATE_LIMIT_DELAY_SECONDS,
        )
