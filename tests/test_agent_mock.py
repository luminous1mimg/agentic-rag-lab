from agentic_rag.agent import run_minimal_agent
from agentic_rag.llm import LLMConfig, build_llm


def test_minimal_agent_mock_runs():
    llm = build_llm(LLMConfig(provider="mock"))
    res = run_minimal_agent("hello", llm, provider="mock", model="mock")
    assert "mock" in res.answer.lower()
    assert "hello" in res.answer.lower() or len(res.answer) > 0
