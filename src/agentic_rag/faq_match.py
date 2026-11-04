"""Deterministic FAQ matching, merged from IndustrialFAQ-RAG.

The original implementation (``IndustrialFAQ-RAG/src/faq_cfd_software/match.py``)
matched a query with ``difflib`` and then reverse-looked-up the standard
question with a substring check (``matched_sent in std``), which could
return the wrong entry. This version keeps the same idea and the same
``text_match`` return shape, but fixes that lookup:

* every candidate sentence (standard question *and* each alias) maps
  exactly to its own entry;
* the best ``difflib`` candidate is resolved through that map, never
  through a substring search;
* malformed TSV lines are skipped instead of crashing the loader;
* the parsed FAQ is cached, so repeated queries do not re-read the file.

Data format (tab-separated, one entry per line)::

    标准问<TAB>相似问1|相似问2<TAB>答案
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_TSV = Path(__file__).resolve().parent / "faq_data" / "faq.tsv"
NO_MATCH = "抱歉，无匹配"


@dataclass(frozen=True)
class FaqEntry:
    standard: str
    aliases: Tuple[str, ...]
    answer: str

    @property
    def source(self) -> str:
        return f"openfoam_faq.tsv:{self.standard}"


@dataclass(frozen=True)
class FaqMatch:
    entry: FaqEntry
    matched_text: str
    score: float

    @property
    def standard(self) -> str:
        return self.entry.standard

    @property
    def answer(self) -> str:
        return self.entry.answer

    @property
    def source(self) -> str:
        return self.entry.source


def _parse_tsv(text: str) -> List[FaqEntry]:
    entries: List[FaqEntry] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) != 3:
            continue
        standard, sims, answer = (p.strip() for p in parts)
        if not standard or not answer:
            continue
        aliases = tuple(a.strip() for a in sims.split("|") if a.strip())
        entries.append(FaqEntry(standard=standard, aliases=aliases, answer=answer))
    return entries


@lru_cache(maxsize=32)
def _load_entries_cached(tsv: str) -> Tuple[FaqEntry, ...]:
    return tuple(_parse_tsv(Path(tsv).read_text(encoding="utf-8")))


def load_faq(tsv: Path = DEFAULT_TSV) -> List[FaqEntry]:
    """Load FAQ entries from a TSV file (cached by path)."""
    return list(_load_entries_cached(str(Path(tsv))))


def match_faq(
    query: str,
    tsv: Path = DEFAULT_TSV,
    threshold: float = 0.6,
) -> Optional[FaqMatch]:
    """Return the best deterministic FAQ match, or ``None``.

    An exact match against a standard question or an alias scores 1.0.
    Otherwise the best ``difflib`` candidate above ``threshold`` wins.
    """
    query = (query or "").strip()
    if not query:
        return None

    entries = load_faq(tsv)
    sentence_to_entry = {}
    for entry in entries:
        sentence_to_entry[entry.standard] = entry
        for alias in entry.aliases:
            sentence_to_entry.setdefault(alias, entry)

    if query in sentence_to_entry:
        entry = sentence_to_entry[query]
        return FaqMatch(entry=entry, matched_text=query, score=1.0)

    candidates = list(sentence_to_entry)
    best = difflib.get_close_matches(query, candidates, n=1, cutoff=threshold)
    if not best:
        return None

    matched_text = best[0]
    entry = sentence_to_entry[matched_text]
    score = difflib.SequenceMatcher(None, query, matched_text).ratio()
    return FaqMatch(entry=entry, matched_text=matched_text, score=score)


def text_match(
    query: str,
    threshold: float = 0.6,
    tsv: Path = DEFAULT_TSV,
) -> Tuple[Optional[str], str]:
    """Backward-compatible wrapper for the original ``match.py`` API.

    Returns ``(standard_question, answer)`` on a hit, or
    ``(None, "抱歉，无匹配")`` on a miss.
    """
    match = match_faq(query, tsv=tsv, threshold=threshold)
    if match is None:
        return None, NO_MATCH
    return match.standard, match.answer
