from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from operator import add
from pathlib import Path
from typing import Annotated, Any, Dict, List, TypedDict

from langgraph.graph import END, START, StateGraph

from .indexer import ensure_index
from .retriever import retrieve
from .rag import build_rag_prompt
from .llm import BaseLLM


# --------- Graph State ---------
class RagGraphState(TypedDict, total=False):
    # inputs
    question: str
    query: str

    # retrieval + answer
    chunks: List[Dict[str, Any]]   # keep dicts for serialization
    answer: str
    sources: List[str]

    # loop control
    tries: int
    max_tries: int
    needs_rewrite: bool

    # memory (accumulated)
    history: Annotated[List[Dict[str, Any]], add]  # reducer: list append


@dataclass(frozen=True)
class GraphResult:
    answer: str
    sources: List[str]


def _dedup_cap_sources(sources: List[str], cap: int = 5) -> List[str]:
    seen = set()
    out = []
    for s in sources:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= cap:
            break
    return out


def _rewrite_query(llm: BaseLLM, question: str, prev_query: str) -> str:
    prompt = (
        "你是检索专家。请把用户问题改写成一个更适合检索的短查询（<= 16 个词/短语）。\n"
        "要求：\n"
        "- 只输出一行 query，不要解释\n"
        "- 保留关键实体/关键词\n"
        "- 不要输出引号\n\n"
        "- 如果知识库主要是英文内容，可以把 query 翻译成英文\n\n"
        f"用户问题：{question}\n"
        f"上一次 query：{prev_query}\n"
        "改写后的 query："
    )
    q = (llm.generate(prompt) or "").strip()
    q = re.sub(r"\s+", " ", q).strip()

    # fallback：防止模型输出一堆解释
    if not q or len(q) > 80:
        return prev_query if prev_query else question

    # 有些模型会输出类似 "query: xxx"
    q = re.sub(r"^(query|Query)\s*:\s*", "", q).strip()
    return q or (prev_query if prev_query else question)


def _build_checkpointer_sqlite(db_path: Path):
    """
    Needs: langgraph-checkpoint-sqlite
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency: langgraph-checkpoint-sqlite. "
            "Run: python -m pip install -U langgraph-checkpoint-sqlite"
        ) from e

    db_path.parent.mkdir(parents=True, exist_ok=True)
    # 官方文档示例提示：check_same_thread=False 可以配合内部锁使用 :contentReference[oaicite:4]{index=4}
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


def reset_thread(db_path: Path, thread_id: str) -> None:
    """
    Clear all checkpoints for a thread_id.
    """
    cp = _build_checkpointer_sqlite(db_path)
    # SqliteSaver 提供 delete_thread 方法 :contentReference[oaicite:5]{index=5}
    cp.delete_thread(thread_id)


def build_rag_graph(
    llm: BaseLLM,
    *,
    kb_dir: Path,
    index_dir: Path,
    reindex: bool = False,
    top_k: int = 5,
    max_tries: int = 2,
    score_threshold: float = 0.05,
):
    """
    Graph: retrieve -> answer -> judge -> (rewrite -> retrieve)* -> end
    """

    def node_retrieve(state: RagGraphState) -> Dict[str, Any]:
        index = ensure_index(kb_dir=kb_dir, index_dir=index_dir, reindex=reindex)
        q = state.get("query") or state.get("question") or ""
        hits = retrieve(q, index, top_k=top_k)

        chunks = [
            {
                "source": h.source,
                "text": h.text,
                "score": h.score,
            }
            for h in hits
        ]
        return {"chunks": chunks}

    def node_answer(state: RagGraphState) -> Dict[str, Any]:
        chunks = state.get("chunks") or []
        question = state.get("question") or ""

        if not chunks:
            ans = "资料不足：当前知识库检索不到相关内容。你希望我从哪些文档/主题中查找？"
            return {
                "answer": ans,
                "sources": [],
                "history": [{"q": question, "query": state.get("query", ""), "a": ans, "sources": []}],
            }

        # build contexts
        retrieved_like = []
        for i, c in enumerate(chunks, start=1):
            retrieved_like.append(type("Tmp", (), {"source": c["source"], "text": c["text"]})())

        prompt = build_rag_prompt(question, retrieved_like)  # reuse your Step2 prompt builder
        ans = (llm.generate(prompt) or "").strip()

        sources = _dedup_cap_sources([c["source"] for c in chunks], cap=5)
        return {
            "answer": ans,
            "sources": sources,
            "history": [{"q": question, "query": state.get("query", ""), "a": ans, "sources": sources}],
        }

    def node_judge(state: RagGraphState) -> Dict[str, Any]:
        chunks = state.get("chunks") or []
        ans = (state.get("answer") or "").strip()
        tries = int(state.get("tries") or 0)

        # Heuristic judge:
        # 1) no chunks -> rewrite
        # 2) answer says "资料不足" -> rewrite
        # 3) low score -> rewrite
        needs = False
        if not chunks:
            needs = True
        elif "资料不足" in ans:
            needs = True
        else:
            top_score = float(chunks[0].get("score", 0.0))
            if top_score < score_threshold:
                needs = True

        # stop if exceeded
        if tries >= int(state.get("max_tries") or max_tries):
            needs = False

        return {"needs_rewrite": needs}

    def node_rewrite(state: RagGraphState) -> Dict[str, Any]:
        question = state.get("question") or ""
        prev_q = state.get("query") or question
        new_q = _rewrite_query(llm, question, prev_q)
        return {"query": new_q, "tries": int(state.get("tries") or 0) + 1}

    def route_after_judge(state: RagGraphState) -> str:
        return "rewrite" if state.get("needs_rewrite") else END

    g = StateGraph(RagGraphState)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer", node_answer)
    g.add_node("judge", node_judge)
    g.add_node("rewrite", node_rewrite)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "judge")
    g.add_conditional_edges("judge", route_after_judge, {"rewrite": "rewrite", END: END})
    g.add_edge("rewrite", "retrieve")

    return g


def run_rag_graph(
    question: str,
    llm: BaseLLM,
    *,
    kb_dir: Path,
    index_dir: Path,
    checkpoint_db: Path,
    thread_id: str,
    reindex: bool = False,
    top_k: int = 5,
    max_tries: int = 2,
) -> GraphResult:
    graph_builder = build_rag_graph(
        llm,
        kb_dir=kb_dir,
        index_dir=index_dir,
        reindex=reindex,
        top_k=top_k,
        max_tries=max_tries,
    )

    checkpointer = _build_checkpointer_sqlite(checkpoint_db)
    graph = graph_builder.compile(checkpointer=checkpointer)

    # thread_id 必须放在 configurable config 里 :contentReference[oaicite:6]{index=6}
    config = {"configurable": {"thread_id": thread_id}}

    init_state: RagGraphState = {
        "question": question,
        "query": question,
        "tries": 0,
        "max_tries": max_tries,
        "history": [],
    }

    out: RagGraphState = graph.invoke(init_state, config=config)
    return GraphResult(answer=out.get("answer", ""), sources=out.get("sources", []))


def get_latest_state(
    *,
    llm: BaseLLM,
    kb_dir: Path,
    index_dir: Path,
    checkpoint_db: Path,
    thread_id: str,
    reindex: bool = False,
    top_k: int = 5,
    max_tries: int = 2,
) -> RagGraphState:
    graph_builder = build_rag_graph(
        llm,
        kb_dir=kb_dir,
        index_dir=index_dir,
        reindex=reindex,
        top_k=top_k,
        max_tries=max_tries,
    )
    checkpointer = _build_checkpointer_sqlite(checkpoint_db)
    graph = graph_builder.compile(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": thread_id}}
    snap = graph.get_state(config)  # get latest snapshot :contentReference[oaicite:7]{index=7}
    return snap.values  # type: ignore[return-value]
