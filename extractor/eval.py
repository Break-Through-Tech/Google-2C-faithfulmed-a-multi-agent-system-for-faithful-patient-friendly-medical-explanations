"""
Extractor evaluation against MedAESQA expert nuggets. No LLM calls.

Nugget format: "[hedge] Predicate (subject [CUI], object [CUI])", e.g.
    "[may] Treat (drugs [C1547857], sleep apnea [C0037315])"
A few nuggets don't fit (IF/THEN, comparisons); they still count toward recall and are
matched on their full text.

Matching (MATCHER_VERSION) is lexical. An atom's tokens are its claim + subject +
predicate + object. A nugget matches only if EVERY argument is at least
MATCH_THRESHOLD covered by those tokens; ranking uses mean coverage. Requiring every
argument (not overall overlap) stops sibling atoms that share subject and predicate
("RUNX2 mutations cause X" vs "... cause Y") from matching each other's nuggets.
Predicate words count because nuggets sometimes put the action in an argument
("Treat (Remove, ovarian cyst)" vs atom "surgery may remove the cyst").

It misses paraphrases (spill vs spillage). Validate by hand on dev examples, and freeze
MATCHER_VERSION before scoring any test item.

Precision is a lower bound: the source can contain true facts the experts did not write
as nuggets (e.g. definitions), and those count as unmatched atoms here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from extractor.schema import Atom, Hedge

_NUGGET = re.compile(r"^\s*(?:\[([^\]]+)\]\s*)?([A-Za-z][A-Za-z _-]*?)\s*\((.*)\)\s*$")
_CUI = re.compile(r"\[C\d+\]")
_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an the of to in on for with and or by at from as is are be it its this that "
    "can may all into if then".split()
)

_HEDGE_MAP: dict[str, Hedge] = {
    "may": "possible", "might": "possible", "can": "possible", "could": "possible",
    "frequently": "frequency", "often": "frequency", "sometimes": "frequency",
    "infrequently": "frequency", "commonly": "frequency", "common": "frequency",
    "most common": "frequency", "most frequent": "frequency", "most": "frequency",
    "mostly": "frequency", "some": "frequency", "rarely": "frequency",
    "should": "recommended", "standard": "recommended", "best": "recommended",
    "recommended": "recommended", "try": "recommended",
    "not": "negated", "no": "negated", "never": "negated",
}

MATCHER_VERSION = "lexical-v2"  # v1: Dice on subject+object only (missed q133 matches)
MATCH_THRESHOLD = 0.5
_COMPLEX = re.compile(r"\b(IF|THEN|LESS|MORE|THAN)\b")  # uppercase operators in nuggets


@dataclass
class Nugget:
    raw: str
    hedge: Hedge | None  # None = hedge word we don't map (e.g. "reported")
    predicate: str | None
    args: list[str]
    parsed: bool


@dataclass
class Matching:
    pairs: list[tuple[Atom, Nugget, float]] = field(default_factory=list)
    unmatched_atoms: list[Atom] = field(default_factory=list)
    unmatched_nuggets: list[Nugget] = field(default_factory=list)


def _split_args(s: str) -> list[str]:
    """Split on commas that are not inside brackets/parentheses."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [_CUI.sub("", p).strip() for p in parts if p.strip()]


def map_hedge(word: str | None) -> Hedge | None:
    if word is None:
        return "asserted"
    return _HEDGE_MAP.get(word.strip().lower())


def parse_nugget(s: str) -> Nugget:
    m = None if _COMPLEX.search(s) else _NUGGET.match(s)
    if not m:
        return Nugget(raw=s, hedge=None, predicate=None, args=[_CUI.sub("", s).strip()],
                      parsed=False)
    hedge_word, predicate, inner = m.groups()
    return Nugget(raw=s, hedge=map_hedge(hedge_word), predicate=predicate.strip().lower(),
                  args=_split_args(inner), parsed=True)


def _tokens(text: str) -> set[str]:
    toks = set()
    for t in _TOKEN.findall(text.lower()):
        if t in _STOPWORDS:
            continue
        toks.add(t[:-1] if len(t) > 3 and t.endswith("s") else t)  # crude plural strip
    return toks


def similarity(atom: Atom, nugget: Nugget, threshold: float = MATCH_THRESHOLD) -> float:
    """Mean per-argument token coverage; 0 if any argument is below threshold."""
    a = _tokens(f"{atom.claim} {atom.subject} {atom.predicate} {atom.object}")
    coverages = []
    for arg in nugget.args:
        n = _tokens(arg)
        if n:
            coverages.append(len(a & n) / len(n))
    if not coverages or min(coverages) < threshold:
        return 0.0
    return sum(coverages) / len(coverages)


def match(atoms: list[Atom], nugget_strings: list[str],
          threshold: float = MATCH_THRESHOLD) -> Matching:
    """Greedy 1:1 assignment, highest similarity first."""
    nuggets = [parse_nugget(s) for s in nugget_strings]
    candidates = sorted(
        ((similarity(a, n, threshold), i, j)
         for i, a in enumerate(atoms) for j, n in enumerate(nuggets)),
        reverse=True,
    )
    used_a, used_n, result = set(), set(), Matching()
    for score, i, j in candidates:
        if score < threshold:
            break
        if i in used_a or j in used_n:
            continue
        used_a.add(i)
        used_n.add(j)
        result.pairs.append((atoms[i], nuggets[j], score))
    result.unmatched_atoms = [a for i, a in enumerate(atoms) if i not in used_a]
    result.unmatched_nuggets = [n for j, n in enumerate(nuggets) if j not in used_n]
    return result


def score(m: Matching, evidence_failures: list[str] | None = None,
          hedge_flags: list[str] | None = None) -> dict:
    n_matched = len(m.pairs)
    n_atoms = n_matched + len(m.unmatched_atoms)
    n_nuggets = n_matched + len(m.unmatched_nuggets)
    hedge_pairs = [(a, n) for a, n, _ in m.pairs if n.hedge is not None]
    return {
        "nugget_recall": n_matched / n_nuggets if n_nuggets else 0.0,
        "atom_precision_lower_bound": n_matched / n_atoms if n_atoms else 0.0,
        "hedge_accuracy": (sum(a.hedge == n.hedge for a, n in hedge_pairs) / len(hedge_pairs)
                           if hedge_pairs else None),
        "evidence_fail_rate": (len(evidence_failures) / n_atoms
                               if evidence_failures is not None and n_atoms else None),
        "hedge_flag_rate": (len(hedge_flags) / n_atoms
                            if hedge_flags is not None and n_atoms else None),
        "n_atoms": n_atoms,
        "n_nuggets": n_nuggets,
        "n_matched": n_matched,
        "matcher_version": MATCHER_VERSION,
    }
