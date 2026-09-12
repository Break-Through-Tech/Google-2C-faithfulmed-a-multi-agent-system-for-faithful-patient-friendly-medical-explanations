from __future__ import annotations

from dataclasses import dataclass
import re

from retrieval.ids import retrieval_chunk_id
from retrieval.schema import (
    NormalizedDocument,
    RetrievalChunk,
)


_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z0-9])"
)


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    """
    Character-based approximation of Gemini token sizes.

    Gemini documentation says roughly 4 characters/token
    for typical text.

    target_chars=2200  ≈ 550 tokens
    max_chars=2800     ≈ 700 tokens
    overlap_chars=320  ≈ 80 tokens
    """

    target_chars: int = 2200
    max_chars: int = 2800
    overlap_chars: int = 320


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


def _join_units(units: list[str]) -> str:
    return "\n\n".join(
        unit.strip()
        for unit in units
        if unit.strip()
    )


def _hard_split_words(
    text: str,
    max_chars: int,
) -> list[str]:
    """
    Last-resort split for text containing an exceptionally long
    sentence or block.

    Never splits inside a word.
    """

    words = text.split()

    if not words:
        return []

    pieces: list[str] = []
    current: list[str] = []
    current_length = 0

    for word in words:

        added_length = len(word)

        if current:
            added_length += 1

        if (
            current
            and current_length + added_length > max_chars
        ):
            pieces.append(" ".join(current))

            current = [word]
            current_length = len(word)

        else:
            current.append(word)
            current_length += added_length

    if current:
        pieces.append(" ".join(current))

    return pieces


def _split_long_block(
    block: str,
    max_chars: int,
) -> list[str]:
    """
    Split an oversized paragraph at sentence boundaries.

    Only falls back to word boundaries when an individual sentence
    itself exceeds max_chars.
    """

    block = block.strip()

    if len(block) <= max_chars:
        return [block]

    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_BOUNDARY_RE.split(block)
        if sentence.strip()
    ]

    # If sentence detection failed, use word-level splitting.
    if len(sentences) <= 1:
        return _hard_split_words(
            block,
            max_chars,
        )

    pieces: list[str] = []
    current: list[str] = []

    for sentence in sentences:

        # Extremely long sentence.
        if len(sentence) > max_chars:

            if current:
                pieces.append(" ".join(current))
                current = []

            pieces.extend(
                _hard_split_words(
                    sentence,
                    max_chars,
                )
            )

            continue

        candidate = (
            " ".join(current + [sentence])
            if current
            else sentence
        )

        if (
            current
            and len(candidate) > max_chars
        ):
            pieces.append(" ".join(current))
            current = [sentence]

        else:
            current.append(sentence)

    if current:
        pieces.append(" ".join(current))

    return pieces


def _prepare_units(
    blocks: list[str],
    max_chars: int,
) -> list[str]:
    """
    Convert source blocks into chunking units.

    Normal paragraphs/lists remain intact.
    Only oversized blocks are split.
    """

    units: list[str] = []

    for block in blocks:

        clean_block = block.strip()

        if not clean_block:
            continue

        units.extend(
            _split_long_block(
                clean_block,
                max_chars,
            )
        )

    return units


def _chunk_units(
    units: list[str],
    config: ChunkingConfig,
) -> list[str]:
    """
    Pack paragraph/sentence units into chunks.

    Overlap occurs only at complete unit boundaries.
    """

    if not units:
        return []

    entire_text = _join_units(units)

    # Do not split short documents merely for the sake of chunking.
    if len(entire_text) <= config.max_chars:
        return [entire_text]

    chunks: list[str] = []

    start = 0

    while start < len(units):

        chunk_units: list[str] = []
        current_chars = 0

        end = start

        while end < len(units):

            unit = units[end]

            separator_chars = (
                2 if chunk_units else 0
            )

            candidate_chars = (
                current_chars
                + separator_chars
                + len(unit)
            )

            # Once we already have content, never exceed the hard limit.
            if (
                chunk_units
                and candidate_chars > config.max_chars
            ):
                break

            chunk_units.append(unit)
            current_chars = candidate_chars
            end += 1

            # Target reached. Stop here unless this was the only unit.
            if current_chars >= config.target_chars:
                break

        chunk_text = _join_units(chunk_units)

        if chunk_text:
            chunks.append(chunk_text)

        if end >= len(units):
            break

        # Choose overlap from complete trailing units.
        overlap_start = end
        overlap_chars = 0

        while overlap_start > start:

            candidate = units[overlap_start - 1]

            additional = (
                len(candidate)
                + (2 if overlap_chars else 0)
            )

            if (
                overlap_chars + additional
                > config.overlap_chars
            ):
                break

            overlap_start -= 1
            overlap_chars += additional

        # Guarantee forward progress.
        if overlap_start <= start:
            start = end
        else:
            start = overlap_start

    return chunks


def chunk_document(
    document: NormalizedDocument,
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> list[RetrievalChunk]:
    """
    Convert one normalized source document into final retrieval units.
    """

    chunks: list[RetrievalChunk] = []

    global_chunk_index = 0

    for section in document.sections:

        # Glossary definitions are already atomic retrieval units.
        if document.source_type == "glossary_definition":

            section_texts = [
                _join_units(section.blocks)
            ]

        else:
            units = _prepare_units(
                section.blocks,
                max_chars=config.max_chars,
            )

            section_texts = _chunk_units(
                units,
                config,
            )

        for text in section_texts:

            if not text:
                continue

            chunk_id = retrieval_chunk_id(
                document.parent_id,
                global_chunk_index,
            )

            chunk = RetrievalChunk(
                id=chunk_id,
                parent_id=document.parent_id,

                title=document.title,
                text=text,

                source=document.source,
                source_type=document.source_type,
                url=document.url,

                section=section.name,
                chunk_index=global_chunk_index,

                language=document.language,

                topic_id=document.topic_id,
                category=document.category,

                synonyms=list(document.synonyms),
                see_references=list(
                    document.see_references
                ),
                mesh_terms=list(
                    document.mesh_terms
                ),

                groups=list(document.groups),
                related_topics=list(
                    document.related_topics
                ),

                primary_institute=(
                    document.primary_institute
                ),

                organization=(
                    document.organization
                ),

                meta_description=(
                    document.meta_description
                ),

                date_created=(
                    document.date_created
                ),

                source_generated_at=(
                    document.source_generated_at
                ),

                source_file=(
                    document.source_file
                ),
            )

            chunks.append(chunk)

            global_chunk_index += 1

    return chunks