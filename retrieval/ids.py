import hashlib
import re
import uuid


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def short_hash(*parts: str, length: int = 10) -> str:
    joined = "\x1f".join(parts)
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
    return digest[:length]


def glossary_parent_id(category: str, term: str) -> str:
    slug = slugify(term)
    digest = short_hash(category, term)

    return f"medlineplus:glossary:{category}:{slug}:{digest}"


def health_topic_parent_id(topic_id: str) -> str:
    return f"medlineplus:health_topic:{topic_id}"


def retrieval_chunk_id(parent_id: str, chunk_index: int) -> str:
    return f"{parent_id}:chunk:{chunk_index:03d}"

# Fixed namespace so every machine maps a chunk ID to the same Qdrant point.
_POINT_ID_NAMESPACE = uuid.UUID("5b0f7d4e-3c1a-4d8e-9f2b-6a7c8d9e0f1a")


def point_id(chunk_id: str) -> str:
    """Qdrant point IDs must be UUIDs or integers; derive a stable UUID."""

    return str(uuid.uuid5(_POINT_ID_NAMESPACE, chunk_id))
