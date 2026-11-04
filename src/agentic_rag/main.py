from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import run_minimal_agent
from .llm import LLMConfig, build_llm


def main() -> int:
    parser = argparse.ArgumentParser(description="agentic-rag-lab CLI")
    parser.add_argument("question", nargs="?", default="", help="Your question/prompt")

    parser.add_argument(
        "--provider",
        choices=["auto", "mock", "openai", "gemini"],
        default="auto",
        help="LLM provider. auto = openai if OPENAI_API_KEY, else gemini if GEMINI_API_KEY, else mock",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", os.getenv("GEMINI_MODEL", "gpt-5-mini")),
        help="Model name (used when provider is openai/gemini/auto)",
    )
    parser.add_argument("--show-meta", action="store_true", help="Print provider/model info")

    # Step2: RAG args
    parser.add_argument("--mode", choices=["chat", "rag", "graph", "faq"], default="chat")
    parser.add_argument("--faq-tsv", default=None, help="FAQ TSV file for --mode faq (default: package faq_data/faq.tsv)")
    parser.add_argument("--faq-threshold", type=float, default=0.6, help="difflib cutoff for --mode faq")
    parser.add_argument("--kb", default="examples/kb")
    parser.add_argument("--index-dir", default="data/index_lexical")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--thread-id", default="demo", help="LangGraph thread id for persistence")
    parser.add_argument("--checkpoint-db", default="data/checkpoints.sqlite", help="SQLite checkpoint db path")
    parser.add_argument("--reset-thread", action="store_true", help="Delete checkpoints for thread_id")
    parser.add_argument("--max-tries", type=int, default=2, help="Max rewrite loops in graph mode")
    parser.add_argument("--show-history", action="store_true", help="Print recent history for this thread")

    args = parser.parse_args()
    if not args.question.strip():
        print('Usage: python -m agentic_rag.main "your question"')
        return 2

    # ---- FAQ branch: deterministic match first, no LLM needed on a hit ----
    if args.mode == "faq":
        from .faq_match import DEFAULT_TSV, match_faq

        faq_tsv = Path(args.faq_tsv) if args.faq_tsv else DEFAULT_TSV
        match = match_faq(args.question, tsv=faq_tsv, threshold=args.faq_threshold)
        if match is not None:
            print(match.answer)
            print("\nSources:")
            print(f"- {match.source}")
            return 0

        # No deterministic hit: fall back to RAG over the OpenFOAM KB
        # when the caller did not point --kb elsewhere.
        kb_dir = Path(args.kb)
        if args.kb == "examples/kb" and (Path("examples/kb/openfoam").is_dir()):
            kb_dir = Path("examples/kb/openfoam")
        cfg = LLMConfig(provider=args.provider, model=args.model)
        llm = build_llm(cfg)
        from .rag import run_rag

        res = run_rag(
            args.question,
            llm,
            kb_dir=kb_dir,
            index_dir=Path(args.index_dir),
            top_k=args.top_k,
            reindex=args.reindex,
        )
        print(res.answer)
        if res.sources:
            print("\nSources:")
            for src in res.sources:
                print(f"- {src}")
        return 0

    cfg = LLMConfig(provider=args.provider, model=args.model)
    llm = build_llm(cfg)

    # ---- RAG branch first (so it actually runs) ----
    if args.mode == "graph":
        from .graph import run_rag_graph, reset_thread, get_latest_state

        if args.reset_thread:
            reset_thread(Path(args.checkpoint_db), args.thread_id)

        res = run_rag_graph(
            args.question,
            llm,
            kb_dir=Path(args.kb),
            index_dir=Path(args.index_dir),
            checkpoint_db=Path(args.checkpoint_db),
            thread_id=args.thread_id,
            reindex=args.reindex,
            top_k=args.top_k,
            max_tries=args.max_tries,
        )

        if args.show_meta:
            print(f"[meta] provider={args.provider} model={args.model}")

        print(res.answer)
        if res.sources:
            print("\nSources:")
            for s in res.sources:
                print(f"- {s}")

        if args.show_history:
            st = get_latest_state(
                llm=llm,
                kb_dir=Path(args.kb),
                index_dir=Path(args.index_dir),
                checkpoint_db=Path(args.checkpoint_db),
                thread_id=args.thread_id,
                reindex=args.reindex,
                top_k=args.top_k,
                max_tries=args.max_tries,
            )
            hist = st.get("history", []) or []
            print("\n[history] last 3:")
            for item in hist[-3:]:
                print(f"- q={item.get('q')}")
        return 0

    if args.mode == "rag":
        from .rag import run_rag

        res = run_rag(
            args.question,
            llm,
            kb_dir=Path(args.kb),
            index_dir=Path(args.index_dir),
            top_k=args.top_k,
            reindex=args.reindex,
        )

        if args.show_meta:
            print(f"[meta] provider={args.provider} model={args.model}")

        print(res.answer)
        if res.sources:
            print("\nSources:")
            for s in res.sources:
                print(f"- {s}")
        return 0

    # ---- chat branch ----
    result = run_minimal_agent(args.question, llm, provider=args.provider, model=args.model)

    if args.show_meta:
        print(f"[meta] provider={result.provider} model={result.model}")

    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
