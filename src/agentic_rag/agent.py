from __future__ import annotations

from dataclasses import dataclass

from .llm import BaseLLM


SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Be concise, correct, and if information is missing, ask one clarifying question."
)


@dataclass
class AgentResult:
    answer: str
    provider: str
    model: str


def run_minimal_agent(question: str, llm: BaseLLM, provider: str, model: str) -> AgentResult:
    prompt = f"{SYSTEM_PROMPT}\n\nUser question:\n{question}\n\nAssistant answer:"
    answer = llm.generate(prompt).strip()
    return AgentResult(answer=answer, provider=provider, model=model)
