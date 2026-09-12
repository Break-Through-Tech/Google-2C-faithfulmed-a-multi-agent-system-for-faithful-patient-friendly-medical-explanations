import hashlib
import re


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