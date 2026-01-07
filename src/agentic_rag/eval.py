from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .llm import LLMConfig, build_llm


@dataclass
class EvalCase:
    id: str
    question: str
    expected_sources: List[str]
    must_include: List[str]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_eval_cases(path: Path) -> List[EvalCase]:
    rows = _load_jsonl(path)
    out: List[EvalCase] = []
    for r in rows:
        out.append(
            EvalCase(
                id=str(r["id"]),
                question=str(r["question"]),
                expected_sources=list(r.get("expected_sources", [])),
                must_include=list(r.get("must_include", [])),
            )
        )
    return out


def normalize_source_file(source: str) -> str:
    # "dev.md#c2" -> "dev.md"
    return source.split("#", 1)[0].strip()


def has_citation_markers(answer: str) -> bool:
    # look for [1] [2] ...
    return bool(re.search(r"\[\d+\]", answer))


def must_include_ok(answer: str, must: List[str]) -> Tuple[bool, List[str]]:
    missing = [m for m in must if m not in answer]
    return (len(missing) == 0, missing)


def expected_sources_hit(pred_sources: List[str], expected_files: List[str]) -> Tuple[bool, List[str]]:
    if not expected_files:
        return True, []
    pred_files = {normalize_source_file(s) for s in pred_sources}
    missing = [e for e in expected_files if e not in pred_files]
    return (len(missing) == 0, missing)


def run_one(
    *,
    mode: str,
    question: str,
    llm,
    kb_dir: Path,
    index_dir: Path,
    top_k: int,
    reindex: bool,
    checkpoint_db: Path,
    thread_id: str,
    max_tries: int,
) -> Tuple[str, List[str], float]:
    t0 = time.perf_counter()

    if mode == "rag":
        from .rag import run_rag

        res = run_rag(
            question,
            llm,
            kb_dir=kb_dir,
            index_dir=index_dir,
            top_k=top_k,
            reindex=reindex,
        )
        ans, sources = res.answer, res.sources

    elif mode == "graph":
        from .graph import run_rag_graph, reset_thread

        # isolate each case
        reset_thread(checkpoint_db, thread_id)
        res = run_rag_graph(
            question,
            llm,
            kb_dir=kb_dir,
            index_dir=index_dir,
            checkpoint_db=checkpoint_db,
            thread_id=thread_id,
            top_k=top_k,
            max_tries=max_tries,
        )
        ans, sources = res.answer, res.sources

    else:
        raise ValueError(f"unknown mode: {mode}")

    dt = time.perf_counter() - t0
    return ans, sources, dt


def write_markdown_report(md_path: Path, summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> None:
    lines = []
    lines.append("# Step3.5 Evaluation Report\n")
    lines.append("## Summary\n")
    for k in [
        "total",
        "pass_rate",
        "source_hit_rate",
        "citation_rate",
        "must_include_rate",
        "avg_latency_s",
    ]:
        if k in summary:
            lines.append(f"- **{k}**: {summary[k]}\n")

    lines.append("\n## Cases\n")
    lines.append("| id | pass | source_hit | citation | must_include | latency(s) |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    for r in rows:
        lines.append(
            f"| {r['id']} | {int(r['pass'])} | {int(r['source_hit'])} | {int(r['citation_ok'])} | {int(r['must_include_ok'])} | {r['latency_s']:.3f} |\n"
        )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluation harness for agentic-rag-lab")
    p.add_argument("--eval", required=True, help="Path to eval_set.jsonl")
    p.add_argument("--out-json", default="reports/step35_baseline.json")
    p.add_argument("--out-md", default="reports/step35_baseline.md")

    p.add_argument("--mode", choices=["rag", "graph"], default="rag")
    p.add_argument("--kb", default="examples/kb")
    p.add_argument("--index-dir", default="data/index_lexical")
    p.add_argument("--reindex", action="store_true")
    p.add_argument("--top-k", type=int, default=5)

    p.add_argument("--provider", choices=["auto", "mock", "openai", "gemini"], default="auto")
    p.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", os.getenv("GEMINI_MODEL", "gpt-5-mini")),
    )
    p.add_argument("--checkpoint-db", default="data/eval_checkpoints.sqlite")
    p.add_argument("--max-tries", type=int, default=2)

    args = p.parse_args()

    eval_path = Path(args.eval)
    kb_dir = Path(args.kb)
    index_dir = Path(args.index_dir)
    checkpoint_db = Path(args.checkpoint_db)

    cfg = LLMConfig(provider=args.provider, model=args.model)
    llm = build_llm(cfg)

    cases = load_eval_cases(eval_path)
    results: List[Dict[str, Any]] = []

    pass_cnt = 0
    source_hit_cnt = 0
    citation_cnt = 0
    must_cnt = 0
    latencies = []

    for c in cases:
        ans, sources, dt = run_one(
            mode=args.mode,
            question=c.question,
            llm=llm,
            kb_dir=kb_dir,
            index_dir=index_dir,
            top_k=args.top_k,
            reindex=args.reindex,
            checkpoint_db=checkpoint_db,
            thread_id=f"eval-{c.id}",
            max_tries=args.max_tries,
        )

        src_hit, missing_src = expected_sources_hit(sources, c.expected_sources)
        cit_ok = has_citation_markers(ans) if sources else True
        must_ok, missing_must = must_include_ok(ans, c.must_include)

        # pass definition (baseline): sources ok + must_include ok
        case_pass = bool(src_hit and must_ok)

        results.append(
            {
                "id": c.id,
                "question": c.question,
                "answer": ans,
                "sources": sources,
                "expected_sources": c.expected_sources,
                "must_include": c.must_include,
                "source_hit": src_hit,
                "missing_expected_sources": missing_src,
                "citation_ok": cit_ok,
                "must_include_ok": must_ok,
                "missing_must_include": missing_must,
                "latency_s": dt,
                "pass": case_pass,
            }
        )

        pass_cnt += int(case_pass)
        source_hit_cnt += int(src_hit)
        citation_cnt += int(cit_ok)
        must_cnt += int(must_ok)
        latencies.append(dt)

    total = len(results)
    summary = {
        "total": total,
        "pass_rate": round(pass_cnt / total, 3) if total else 0.0,
        "source_hit_rate": round(source_hit_cnt / total, 3) if total else 0.0,
        "citation_rate": round(citation_cnt / total, 3) if total else 0.0,
        "must_include_rate": round(must_cnt / total, 3) if total else 0.0,
        "avg_latency_s": round(sum(latencies) / total, 3) if total else 0.0,
        "mode": args.mode,
        "provider": args.provider,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.out_md)
    write_markdown_report(out_md, summary, results)

    print(f"[eval] wrote {out_json}")
    print(f"[eval] wrote {out_md}")
    print(f"[eval] summary: {json.dumps(summary, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
