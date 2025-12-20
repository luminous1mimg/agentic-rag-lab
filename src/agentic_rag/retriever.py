from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .indexer import LexicalIndex, tokenize


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    source: str
    text: str
    score: float


def retrieve(query: str, index: LexicalIndex, top_k: int = 5) -> List[RetrievedChunk]:
    vocab_index = {t: i for i, t in enumerate(index.vocab)}
    toks = tokenize(query)
    if not toks:
        return []

    q = np.zeros(len(index.vocab), dtype=np.float32)
    counts = {}
    for t in toks:
        j = vocab_index.get(t)
        if j is None:
            continue
        counts[j] = counts.get(j, 0) + 1

    if not counts:
        return []

    denom = float(len(toks))
    for j, c in counts.items():
        tf = c / denom
        q[j] = float(tf) * float(index.idf[j])

    q = q / (np.linalg.norm(q) + 1e-12)

    scores = index.matrix @ q  # cosine similarity
    k = min(top_k, len(scores))
    top_idx = np.argsort(-scores)[:k]

    out: List[RetrievedChunk] = []
    for i in top_idx:
        m = index.meta[int(i)]
        out.append(
            RetrievedChunk(
                chunk_id=m["chunk_id"],
                source=f'{m["source"]}#c{m["file_chunk_id"]}',
                text=m["text"],
                score=float(scores[int(i)]),
            )
        )
    return out
