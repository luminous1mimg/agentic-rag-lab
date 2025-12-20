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
