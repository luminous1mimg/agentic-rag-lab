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
    parser.add_argument("--mode", choices=["chat", "rag"], default="chat")
    parser.add_argument("--kb", default="examples/kb")
    parser.add_argument("--index-dir", default="data/index_lexical")
    parser.add_argument("--reindex", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)

    args = parser.parse_args()
    if not args.question.strip():
        print('Usage: python -m agentic_rag.main "your question"')
        return 2

    cfg = LLMConfig(provider=args.provider, model=args.model)
    llm = build_llm(cfg)

    # ---- RAG branch first (so it actually runs) ----
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
