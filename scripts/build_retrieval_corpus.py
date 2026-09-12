from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from retrieval.chunking import (
    DEFAULT_CHUNKING_CONFIG,
    chunk_document,
)
from retrieval.schema import RetrievalChunk
from retrieval.sources.medlineplus_definitions import (
    parse_medlineplus_definitions,
)
from retrieval.sources.medlineplus_topics import (
    parse_medlineplus_health_topics,
)


GLOSSARY_SOURCES = [
    {
        "filename": "fitnessdefinitions.xml",
        "category": "fitness",
        "url": (
            "https://medlineplus.gov/definitions/"
            "fitnessdefinitions.html"
        ),
    },
    {
        "filename": "generalhealthdefinitions.xml",
        "category": "general_health",
        "url": (
            "https://medlineplus.gov/definitions/"
            "generalhealthdefinitions.html"
        ),
    },
    {
        "filename": "mineralsdefinitions.xml",
        "category": "minerals",
        "url": (
            "https://medlineplus.gov/definitions/"
            "mineralsdefinitions.html"
        ),
    },
    {
        "filename": "nutritiondefinitions.xml",
        "category": "nutrition",
        "url": (
            "https://medlineplus.gov/definitions/"
            "nutritiondefinitions.html"
        ),
    },
    {
        "filename": "vitaminsdefinitions.xml",
        "category": "vitamins",
        "url": (
            "https://medlineplus.gov/definitions/"
            "vitaminsdefinitions.html"
        ),
    },
]


def write_jsonl(
    chunks: list[RetrievalChunk],
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file_obj:

        for chunk in chunks:

            file_obj.write(
                json.dumps(
                    chunk.to_dict(),
                    ensure_ascii=False,
                )
            )

            file_obj.write("\n")


def validate_chunks(
    chunks: list[RetrievalChunk],
) -> None:

    ids = [
        chunk.id
        for chunk in chunks
    ]

    duplicate_count = (
        len(ids) - len(set(ids))
    )

    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate chunk IDs."
        )

    empty_text = [
        chunk.id
        for chunk in chunks
        if not chunk.text.strip()
    ]

    if empty_text:
        raise ValueError(
            f"Found {len(empty_text)} empty chunks."
        )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--definitions-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--health-topics",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/retrieval_chunks.jsonl"
        ),
    )

    args = parser.parse_args()

    chunks: list[RetrievalChunk] = []

    documents_loaded = 0

    # ---------------------------------------------------------
    # MedlinePlus lay-language glossary
    # ---------------------------------------------------------

    for spec in GLOSSARY_SOURCES:

        xml_path = (
            args.definitions_dir
            / spec["filename"]
        )

        if not xml_path.exists():
            raise FileNotFoundError(
                f"Missing glossary XML: {xml_path}"
            )

        documents = (
            parse_medlineplus_definitions(
                xml_path,
                category=spec["category"],
                canonical_url=spec["url"],
            )
        )

        documents_loaded += len(documents)

        for document in documents:
            chunks.extend(
                chunk_document(document)
            )

    # ---------------------------------------------------------
    # MedlinePlus Health Topics
    # ---------------------------------------------------------

    health_topic_documents = (
        parse_medlineplus_health_topics(
            args.health_topics
        )
    )

    documents_loaded += len(
        health_topic_documents
    )

    for document in health_topic_documents:
        chunks.extend(
            chunk_document(document)
        )

    validate_chunks(chunks)

    write_jsonl(
        chunks,
        args.output,
    )

    source_type_counts = Counter(
        chunk.source_type
        for chunk in chunks
    )

    print()
    print("FaithfulMed retrieval corpus build")
    print("---------------------------------")
    print(f"Documents loaded: {documents_loaded}")
    print(f"Retrieval chunks: {len(chunks)}")

    for source_type, count in sorted(
        source_type_counts.items()
    ):
        print(
            f"  {source_type}: {count}"
        )

    print()
    print(
        "Chunking configuration:"
    )

    print(
        f"  target chars: "
        f"{DEFAULT_CHUNKING_CONFIG.target_chars}"
    )

    print(
        f"  max chars: "
        f"{DEFAULT_CHUNKING_CONFIG.max_chars}"
    )

    print(
        f"  overlap chars: "
        f"{DEFAULT_CHUNKING_CONFIG.overlap_chars}"
    )

    print()
    print(
        f"Wrote corpus to: {args.output}"
    )


if __name__ == "__main__":
    main()