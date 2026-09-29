from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Sequence

from retrieval.retriever import RetrievalResult, get_default_retriever


DEFAULT_GOLD_PATH = Path("data/eval/retrieval_gold_v1.jsonl")
DEFAULT_K_VALUES = (1, 3, 5)
DEFAULT_MAX_K = 10


@dataclass(slots=True)
class GoldQuery:
    """One hand-labeled query and the documents that answer it."""

    query: str
    relevant_titles: list[str]
    relevant_parent_ids: list[str] = field(default_factory=list)
    notes: str = ""


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def load_gold(gold_path: Path) -> list[GoldQuery]:
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold set does not exist: {gold_path}")

    gold: list[GoldQuery] = []

    for line_number, line in enumerate(
        gold_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            item = GoldQuery(**json.loads(line))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(
                f"Invalid gold query on line {line_number} of "
                f"{gold_path}: {error}"
            ) from error

        if not item.query.strip():
            raise ValueError(f"Gold query on line {line_number} is empty.")

        if not item.relevant_titles and not item.relevant_parent_ids:
            raise ValueError(
                f"Gold query on line {line_number} has no relevant "
                "titles or parent IDs."
            )

        gold.append(item)

    if not gold:
        raise ValueError(f"Gold set contains no queries: {gold_path}")

    return gold


def missing_gold_titles(
    gold: Sequence[GoldQuery],
    corpus_path: Path,
) -> list[str]:
    """Return labeled titles that no corpus chunk carries (likely typos)."""

    corpus_titles: set[str] = set()

    with corpus_path.open(encoding="utf-8") as file_obj:
        for line in file_obj:
            if line.strip():
                corpus_titles.add(_normalize(json.loads(line)["title"]))

    missing: list[str] = []

    for item in gold:
        for title in item.relevant_titles:
            if (
                _normalize(title) not in corpus_titles
                and title not in missing
            ):
                missing.append(title)

    return missing


def is_relevant(result: RetrievalResult, item: GoldQuery) -> bool:
    relevant_titles = {_normalize(title) for title in item.relevant_titles}

    return (
        _normalize(result.title) in relevant_titles
        or result.parent_id in item.relevant_parent_ids
    )


def first_relevant_rank(
    results: Sequence[RetrievalResult],
    item: GoldQuery,
) -> int | None:
    """1-based rank of the first relevant result, or None if absent."""

    for rank, result in enumerate(results, start=1):
        if is_relevant(result, item):
            return rank

    return None


def recall_at_k(ranks: Sequence[int | None], k: int) -> float:
    """
    Share of queries with at least one relevant result in the top k.

    Relevance is labeled per document, so a query counts as recalled as
    soon as any chunk of a relevant document appears.
    """

    if not ranks:
        return 0.0

    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: Sequence[int | None]) -> float:
    if not ranks:
        return 0.0

    return sum(1.0 / rank for rank in ranks if rank is not None) / len(ranks)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate FaithfulMed retrieval on a hand-labeled set."
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        help=(
            "Optional retrieval_chunks.jsonl used to warn about labeled "
            "titles that are not in the corpus."
        ),
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=list(DEFAULT_K_VALUES),
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=DEFAULT_MAX_K,
        help="Results retrieved per query; MRR is computed over these.",
    )
    parser.add_argument(
        "--source-type",
        help="Restrict retrieval to one source_type.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON file for per-query ranks and top results.",
    )
    args = parser.parse_args()

    gold = load_gold(args.gold)

    if args.corpus is not None:
        missing = missing_gold_titles(gold, args.corpus)

        for title in missing:
            print(f"Warning: gold title not found in corpus: {title!r}")

        if missing:
            print()

    retrieve_k = max(args.max_k, *args.k)
    retriever = get_default_retriever()

    ranks: list[int | None] = []
    per_query: list[dict[str, object]] = []

    for item in gold:
        results = retriever.retrieve(
            item.query,
            retrieve_k,
            source_type=args.source_type,
        )
        rank = first_relevant_rank(results, item)
        ranks.append(rank)
        per_query.append({
            "query": item.query,
            "relevant_titles": item.relevant_titles,
            "rank": rank,
            "top_titles": [result.title for result in results[:5]],
        })

    print("FaithfulMed retrieval evaluation")
    print("--------------------------------")
    print(f"Gold set: {args.gold} ({len(gold)} queries)")
    print(f"Source type: {args.source_type or 'all'}")
    print()

    metrics: dict[str, float] = {}

    for k in sorted(set(args.k)):
        metrics[f"recall@{k}"] = recall_at_k(ranks, k)

    metrics[f"mrr@{retrieve_k}"] = mean_reciprocal_rank(ranks)

    for name, value in metrics.items():
        print(f"  {name:<10} {value:.3f}")

    misses = [
        entry for entry in per_query
        if entry["rank"] is None or entry["rank"] > max(args.k)
    ]

    if misses:
        print()
        print(f"Queries without a relevant hit in the top {max(args.k)}:")

        for entry in misses:
            print(f"  - {entry['query']}")
            print(f"      expected: {entry['relevant_titles']}")
            print(f"      got:      {entry['top_titles']}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {"metrics": metrics, "queries": per_query},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print()
        print(f"Wrote per-query results to: {args.output}")


if __name__ == "__main__":
    main()
