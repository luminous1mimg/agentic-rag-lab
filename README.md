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
