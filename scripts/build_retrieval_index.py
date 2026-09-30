from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Iterator, Sequence, TypeVar

from retrieval.embeddings import (
    DEFAULT_EMBEDDING_DIMENSIONS,
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingConfig,
    GeminiEmbeddingClient,
)
from retrieval.index import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_ENV_PATH,
    QdrantIndex,
)
from retrieval.schema import RetrievalChunk


DEFAULT_CORPUS_PATH = Path(
    "data/processed/retrieval_chunks.jsonl"
)
DEFAULT_BATCH_SIZE = 100
CORPUS_READ_ATTEMPTS = 3
CORPUS_READ_RETRY_SECONDS = 2.0

T = TypeVar("T")


def read_corpus_text(
    corpus_path: Path,
    *,
    attempts: int = CORPUS_READ_ATTEMPTS,
    retry_seconds: float = CORPUS_READ_RETRY_SECONDS,
) -> str:
    """Read a corpus, retrying macOS cloud-storage hydration timeouts."""

    if attempts <= 0:
        raise ValueError("Corpus read attempts must be positive.")

    for attempt in range(1, attempts + 1):
        try:
            return corpus_path.read_text(encoding="utf-8")
        except TimeoutError as error:
            if attempt == attempts:
                raise RuntimeError(
                    f"Timed out reading {corpus_path}. The file may be "
                    "stored only in iCloud. In Finder, right-click the file "
                    "or project folder and choose 'Keep Downloaded', then "
                    "run the command again."
                ) from error

            print(
                f"Timed out reading corpus; retrying "
                f"({attempt}/{attempts})..."
            )
            time.sleep(retry_seconds)

    raise AssertionError("Unreachable corpus read state.")


def load_chunks(corpus_path: Path) -> list[RetrievalChunk]:
    """Load and validate retrieval chunks before spending embedding calls."""

    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus does not exist: {corpus_path}")

    chunks: list[RetrievalChunk] = []

    corpus_text = read_corpus_text(corpus_path)

    for line_number, line in enumerate(
        corpus_text.splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            values = json.loads(line)
            chunk = RetrievalChunk(**values)
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                f"Invalid retrieval chunk on line {line_number} of "
                f"{corpus_path}: {error}"
            ) from error

        if not chunk.id.strip():
            raise ValueError(
                f"Chunk on line {line_number} has an empty ID."
            )

        if not chunk.text.strip():
            raise ValueError(
                f"Chunk {chunk.id!r} has empty text."
            )

        chunks.append(chunk)

    if not chunks:
        raise ValueError(f"Corpus contains no chunks: {corpus_path}")

    ids = [chunk.id for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Corpus contains duplicate chunk IDs.")

    return chunks


def batched(
    values: Sequence[T],
    batch_size: int,
) -> Iterator[Sequence[T]]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]


def build_index(
    chunks: Sequence[RetrievalChunk],
    *,
    index: QdrantIndex,
    embedding_client: GeminiEmbeddingClient,
    batch_size: int = DEFAULT_BATCH_SIZE,
    skip_existing: bool = True,
) -> int:
    """Embed and upsert every chunk, returning the final collection count."""

    processed = 0
    total = len(chunks)

    for batch in batched(chunks, batch_size):
        pending_batch = batch

        if skip_existing:
            existing_ids = index.existing_ids([
                chunk.id for chunk in batch
            ])
            pending_batch = [
                chunk
                for chunk in batch
                if chunk.id not in existing_ids
            ]
            processed += len(existing_ids)

        if pending_batch:
            embeddings = embedding_client.embed_documents(pending_batch)
            processed += index.upsert(pending_batch, embeddings)

        print(f"Indexed {processed}/{total} chunks")

    return index.count()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the shared FaithfulMed Qdrant index."
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_PATH,
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION_NAME,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_EMBEDDING_MODEL,
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_EMBEDDING_DIMENSIONS,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--rebuild-existing",
        action="store_true",
        help=(
            "Re-embed and upsert records already present in Qdrant. "
            "By default, existing IDs are skipped so interrupted builds resume."
        ),
    )
    args = parser.parse_args()

    embedding_config = EmbeddingConfig(
        model=args.model,
        dimensions=args.dimensions,
    )
    print(f"Loading corpus: {args.corpus}")
    chunks = load_chunks(args.corpus)

    index = QdrantIndex(
        collection_name=args.collection,
        embedding_config=embedding_config,
        env_path=args.env_file,
    )
    embedding_client = GeminiEmbeddingClient(
        config=embedding_config
    )

    print("FaithfulMed retrieval index build")
    print("--------------------------------")
    print(f"Corpus chunks: {len(chunks)}")
    print(f"Model: {embedding_config.model}")
    print(f"Dimensions: {embedding_config.dimensions}")
    print(f"Collection: {args.collection}")
    print("Vector store: Qdrant")
    print()

    final_count = build_index(
        chunks,
        index=index,
        embedding_client=embedding_client,
        batch_size=args.batch_size,
        skip_existing=not args.rebuild_existing,
    )

    print()
    print(f"Collection records: {final_count}")


if __name__ == "__main__":
    main()
