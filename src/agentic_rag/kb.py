from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class Chunk:
    chunk_id: int       # 全局编号（内部用，保留）
    source: str         # 相对路径，如 intro.md
    file_chunk_id: int  # 文件内编号（新加，引用用它）
    text: str


def iter_kb_files(kb_dir: Path, exts: Tuple[str, ...] = (".md", ".txt")) -> Iterable[Path]:
    for p in sorted(kb_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    """
    Very robust chunking:
    - split by blank lines first
    - then sliding window by characters
    Works for Chinese/English without extra deps.
    """
    text = text.strip()
    if not text:
        return []

    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    chunks: List[str] = []

    for para in paras:
        if len(para) <= chunk_size:
            chunks.append(para)
            continue

        # sliding window
        start = 0
        while start < len(para):
            end = min(len(para), start + chunk_size)
            chunks.append(para[start:end].strip())
            if end >= len(para):
                break
            start = max(0, end - overlap)

    return [c for c in chunks if c]


def load_kb_chunks(kb_dir: Path, chunk_size: int = 900, overlap: int = 120) -> List[Chunk]:
    kb_dir = kb_dir.resolve()
    if not kb_dir.exists():
        raise FileNotFoundError(f"kb_dir not found: {kb_dir}")

    out: List[Chunk] = []
    cid = 0
    for fp in iter_kb_files(kb_dir):
        rel = str(fp.relative_to(kb_dir))
        text = read_text(fp)

        pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for file_cid, piece in enumerate(pieces):  # ⭐文件内编号从 0 开始
            out.append(Chunk(chunk_id=cid, source=rel, file_chunk_id=file_cid, text=piece))
            cid += 1

    return out

