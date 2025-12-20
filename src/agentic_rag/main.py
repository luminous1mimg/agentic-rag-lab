from __future__ import annotations

import argparse
import os

from .agent import run_minimal_agent
from .llm import LLMConfig, build_llm


def main() -> int:
    parser = argparse.ArgumentParser(description="agentic-rag-lab CLI (Step1 minimal agent)")
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

    args = parser.parse_args()
    if not args.question.strip():
        print("Usage: python -m agentic_rag.main \"your question\"")
        return 2

    cfg = LLMConfig(provider=args.provider, model=args.model)
    llm = build_llm(cfg)

    result = run_minimal_agent(args.question, llm, provider=args.provider, model=args.model)

    if args.show_meta:
        print(f"[meta] provider={result.provider} model={result.model}")

    print(result.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
