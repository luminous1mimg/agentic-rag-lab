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
- `--mode`: `chat` or `rag` (default: `chat`)
- `--provider`: `auto|mock|openai|gemini` (default: `auto`)
- `--model`: model name for OpenAI/Gemini (default: `gpt-5-mini`)
- `--kb`: knowledge base directory (default: `examples/kb`)
- `--index-dir`: index output dir (default: `data/index_lexical`)
- `--reindex`: force rebuild index
- `--top-k`: number of chunks to retrieve (default: `5`)
- `--show-meta`: print provider/model metadata

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
