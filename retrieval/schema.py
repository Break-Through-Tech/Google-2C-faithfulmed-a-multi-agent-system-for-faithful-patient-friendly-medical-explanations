from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentSection:
    """
    A logical section inside a normalized source document.

    Examples:
      - MedlinePlus glossary: "definition"
      - MedlinePlus Health Topic: "full_summary"
      - Future CDC/NIH guideline: "Symptoms", "Treatment", etc.
    """

    name: str
    blocks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class NormalizedDocument:
    """
    Source-independent representation produced by source parsers.

    This object is NOT embedded directly.

    Raw XML/HTML/etc. is converted into this structure first.
    Chunking then converts this into RetrievalChunk objects.
    """

    parent_id: str

    title: str
    sections: list[DocumentSection]

    source: str
    source_type: str
    url: str

    language: str = "English"

    # Source-specific identifiers / taxonomy
    topic_id: str | None = None
    category: str | None = None

    # Vocabulary that can help retrieval
    synonyms: list[str] = field(default_factory=list)
    see_references: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)

    # Metadata useful for filtering / attribution
    groups: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)

    primary_institute: str | None = None
    organization: str | None = None

    meta_description: str | None = None

    date_created: str | None = None
    source_generated_at: str | None = None
    source_file: str | None = None


@dataclass(slots=True)
class RetrievalChunk:
    """
    Final retrieval unit.

    One RetrievalChunk corresponds to one vector in ChromaDB.
    """

    id: str
    parent_id: str

    title: str
    text: str

    source: str
    source_type: str
    url: str

    section: str
    chunk_index: int

    language: str = "English"

    topic_id: str | None = None
    category: str | None = None

    synonyms: list[str] = field(default_factory=list)
    see_references: list[str] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)

    groups: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)

    primary_institute: str | None = None
    organization: str | None = None

    meta_description: str | None = None

    date_created: str | None = None
    source_generated_at: str | None = None
    source_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Used when writing retrieval_chunks.jsonl."""
        return asdict(self)

    def to_chroma_metadata(self) -> dict[str, Any]:
        """
        Convert provenance fields into Chroma-safe metadata.

        Empty arrays and None values are deliberately omitted.
        """

        metadata: dict[str, Any] = {
            "parent_id": self.parent_id,
            "title": self.title,
            "source": self.source,
            "source_type": self.source_type,
            "url": self.url,
            "section": self.section,
            "chunk_index": self.chunk_index,
            "language": self.language,
        }

        optional_scalars = {
            "topic_id": self.topic_id,
            "category": self.category,
            "primary_institute": self.primary_institute,
            "organization": self.organization,
            "meta_description": self.meta_description,
            "date_created": self.date_created,
            "source_generated_at": self.source_generated_at,
            "source_file": self.source_file,
        }

        for key, value in optional_scalars.items():
            if value is not None and value != "":
                metadata[key] = value

        optional_arrays = {
            "synonyms": self.synonyms,
            "see_references": self.see_references,
            "mesh_terms": self.mesh_terms,
            "groups": self.groups,
            "related_topics": self.related_topics,
        }

        for key, values in optional_arrays.items():
            if values:
                metadata[key] = values

        return metadata