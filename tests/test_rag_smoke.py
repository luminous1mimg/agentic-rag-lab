from pathlib import Path

from agentic_rag.indexer import ensure_index
from agentic_rag.retriever import retrieve


def test_rag_index_and_retrieve(tmp_path: Path):
    kb = tmp_path / "kb"
    kb.mkdir()
    (kb / "a.md").write_text("苹果是一种水果。它常见颜色有红色和绿色。", encoding="utf-8")
    (kb / "b.md").write_text("香蕉也是一种水果，富含钾元素。", encoding="utf-8")

    index_dir = tmp_path / "index"
    index = ensure_index(kb_dir=kb, index_dir=index_dir, reindex=True)

    hits = retrieve("苹果是什么", index, top_k=3)
    assert len(hits) > 0
    assert "a.md" in hits[0].source
