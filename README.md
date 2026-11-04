# agentic-rag-lab

A reproducible Agentic RAG project skeleton (LangGraph-ready).

## Quickstart
```bash
python -m venv .venv
# activate it, then:
pip install -U pip
pip install -e ".[dev]"
pytest -q
python -m agentic_rag.main "hello"
```

## Use Gemini
```bash
pip install -e ".[dev,gemini]"
export GEMINI_API_KEY="your_api_key"
export GEMINI_MODEL="gemini-2.0-flash"  # optional
python -m agentic_rag.main --provider gemini "hello"
```

## Demo

### Chat (LLM only)
```bash
python -m agentic_rag.main "hi" --mode chat --provider gemini
```

### RAG (KB + retrieval)
```bash
# basic RAG run with the sample KB
python -m agentic_rag.main "hi" --mode rag --kb examples/kb --reindex --provider gemini
# use the sample KB in examples/kb and build a local index
python -m agentic_rag.main "What is this project?" --mode rag --kb examples/kb --provider gemini --reindex

python -m agentic_rag.main "这个项目有哪些模块？" --mode rag --kb examples/kb --reindex --provider gemini

python -m agentic_rag.main "这个项目有哪些模块？" --mode graph --kb examples/kb --provider gemini --thread-id demo1 --reset-thread --show-history

# subsequent runs can omit --reindex
python -m agentic_rag.main "如何扩展知识库？" --mode rag --kb examples/kb --provider gemini
```

## CLI Options
- `--mode`: `chat`, `rag`, `graph` or `faq` (default: `chat`)
- `--provider`: `auto|mock|openai|gemini` (default: `auto`)
- `--model`: model name for OpenAI/Gemini (default: `gpt-5-mini`)
- `--kb`: knowledge base directory (default: `examples/kb`)
- `--index-dir`: index output dir (default: `data/index_lexical`)
- `--reindex`: force rebuild index
- `--top-k`: number of chunks to retrieve (default: `5`)
- `--show-meta`: print provider/model metadata
- `--faq-tsv`: FAQ TSV file for `--mode faq` (default: package `faq_data/faq.tsv`)
- `--faq-threshold`: difflib cutoff for `--mode faq` (default: `0.7`)
- `--max-tries`: max rewrite loops in `graph` mode (default: `2`)

## Providers
- `auto`: OpenAI if `OPENAI_API_KEY` set; else Gemini if `GEMINI_API_KEY` set; else `mock`
- `mock`: no network, always available
- OpenAI: `pip install -e ".[dev,openai]"` and set `OPENAI_API_KEY`
- Gemini: `pip install -e ".[dev,gemini]"` and set `GEMINI_API_KEY`
- Model env overrides: `OPENAI_MODEL`, `GEMINI_MODEL`

## KB & Indexing Notes
- KB files: `.md` / `.txt`, loaded recursively from `--kb`
- Index is stored under `--index-dir`; re-run with `--reindex` after KB changes
- If query language mismatches KB language, the query is translated and retried automatically

## Industrial FAQ (merged from IndustrialFAQ-RAG)

50 OpenFOAM FAQ entries, merged as a deterministic first pass plus a
Markdown KB for general RAG:

```bash
# Deterministic match first; falls back to RAG when there is no hit
python -m agentic_rag.main "怎么生成网格" --mode faq

# General RAG over the same OpenFOAM KB
python -m agentic_rag.main "怎么生成网格" --mode rag --kb examples/kb/openfoam --reindex
```

Details: `docs/industrial-faq-merge.md`, matcher: `src/agentic_rag/faq_match.py`,
data: `src/agentic_rag/faq_data/faq.tsv`, KB: `examples/kb/openfoam/faq.md`.
