from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .indexer import ensure_index
from .llm import BaseLLM
from .retriever import RetrievedChunk, retrieve


@dataclass(frozen=True)
class RagResult:
    answer: str
    sources: List[str]


def build_rag_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    ctx_lines = []
    for i, c in enumerate(chunks, start=1):
        ctx_lines.append(f"[{i}] SOURCE: {c.source}\n{c.text}")

    context_block = "\n\n".join(ctx_lines)

    return (
        "你是一个严谨的助手。只能依据【Context】回答，不要编造。\n"
        "规则：\n1) 只要 Context 非空，就必须先基于 Context 给出“尽可能完整的回答”，不要直接说“资料不足”。\n"
        "2) 如果 Context 只能覆盖部分问题：先回答已覆盖部分，然后输出“需要补充的信息：”并追问 1 个最关键的问题。\n"
        "3) 只有当 Context 为空或明显与问题无关时，才允许回答“资料不足”。\n"
        "4) 每个关键结论句末尾必须标注引用，如 [1][2]。\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:\n"
    )


def run_rag(
    question: str,
    llm: BaseLLM,
    *,
    kb_dir: Path,
    index_dir: Path,
    top_k: int = 5,
    reindex: bool = False,
) -> RagResult:
    index = ensure_index(kb_dir=kb_dir, index_dir=index_dir, reindex=reindex)
    chunks = retrieve(question, index, top_k=top_k)

    if not chunks:
        return RagResult(
            answer="资料不足：在当前知识库中没有检索到相关内容。你希望我从哪些文档/主题中查找？",
            sources=[],
        )

    prompt = build_rag_prompt(question, chunks)
    answer = llm.generate(prompt).strip()
    seen = set()
    sources = []
    for c in chunks:
        if c.source in seen:
            continue
        seen.add(c.source)
        sources.append(c.source)
        if len(sources) >= 5:
            break
    return RagResult(answer=answer, sources=sources)
