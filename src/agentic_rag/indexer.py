from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .kb import Chunk, load_kb_chunks


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer:
    - English words/numbers as tokens
    - Chinese characters as tokens
    This keeps dependencies at 0 and works decently for demo-scale KB.
    """
    text = text.lower()
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)


@dataclass
class LexicalIndex:
    vocab: List[str]
    idf: np.ndarray          # (V,)
    matrix: np.ndarray       # (N, V) normalized tf-idf
    meta: List[Dict]         # length N


def _build_tfidf(chunks: List[Chunk], vocab_size: int = 8000) -> LexicalIndex:
    docs_tokens: List[List[str]] = [tokenize(c.text) for c in chunks]
    n_docs = len(docs_tokens)

    # doc frequency
    df: Dict[str, int] = {}
    for toks in docs_tokens:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    # pick top vocab by df
    items = sorted(df.items(), key=lambda x: (-x[1], x[0]))
    vocab = [t for t, _ in items[:vocab_size]]
    vocab_index = {t: i for i, t in enumerate(vocab)}

    # idf
    idf = np.zeros(len(vocab), dtype=np.float32)
    for t, i in vocab_index.items():
        dfi = df.get(t, 0)
        # smooth idf
        idf[i] = float(math.log((1.0 + n_docs) / (1.0 + dfi)) + 1.0)

    # tf-idf matrix
    mat = np.zeros((n_docs, len(vocab)), dtype=np.float32)

    for doc_i, toks in enumerate(docs_tokens):
        if not toks:
            continue
        counts: Dict[int, int] = {}
        for t in toks:
            j = vocab_index.get(t)
            if j is None:
                continue
            counts[j] = counts.get(j, 0) + 1

        denom = float(len(toks))
        for j, c in counts.items():
            tf = c / denom
            mat[doc_i, j] = float(tf) * idf[j]

    # normalize rows (cosine)
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat = mat / norms
    meta = [
        {
            "chunk_id": c.chunk_id,
            "source": c.source,
            "file_chunk_id": c.file_chunk_id,
            "text": c.text,
        }
        for c in chunks
    ]

    return LexicalIndex(vocab=vocab, idf=idf, matrix=mat, meta=meta)


def save_index(index: LexicalIndex, index_dir: Path) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "vocab.json").write_text(json.dumps(index.vocab, ensure_ascii=False), encoding="utf-8")
    np.save(index_dir / "idf.npy", index.idf)
    np.save(index_dir / "matrix.npy", index.matrix)

    with (index_dir / "meta.jsonl").open("w", encoding="utf-8") as f:
        for row in index.meta:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_index(index_dir: Path) -> LexicalIndex:
    vocab = json.loads((index_dir / "vocab.json").read_text(encoding="utf-8"))
    idf = np.load(index_dir / "idf.npy")
    matrix = np.load(index_dir / "matrix.npy")

    meta: List[Dict] = []
    with (index_dir / "meta.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    if meta and "file_chunk_id" not in meta[0]:
        raise ValueError(
            "Index meta is outdated (missing file_chunk_id). "
            "Please rebuild index with --reindex or delete index dir."
        )
    return LexicalIndex(vocab=vocab, idf=idf, matrix=matrix, meta=meta)


def ensure_index(
    kb_dir: Path,
    index_dir: Path,
    *,
    reindex: bool = False,
    chunk_size: int = 900,
    overlap: int = 120,
    vocab_size: int = 8000,
) -> LexicalIndex:
    need = reindex or not (index_dir / "matrix.npy").exists()
    if not need:
        return load_index(index_dir)

    chunks = load_kb_chunks(kb_dir, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        raise ValueError(f"No documents found in kb_dir: {kb_dir}")

    index = _build_tfidf(chunks, vocab_size=vocab_size)
    save_index(index, index_dir)
    return index
