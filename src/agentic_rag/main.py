from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(description="agentic-rag-lab CLI (Step0 placeholder)")
    parser.add_argument("question", nargs="?", default="", help="Your question/prompt")
    args = parser.parse_args()

    if args.question:
        print(f"[Step0] Received question: {args.question}")
    else:
        print("[Step0] Repo skeleton is ready. Next: Step1 minimal agent.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
