from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Optional


ProviderName = Literal["auto", "mock", "openai", "gemini"]


class LLMError(RuntimeError):
    """LLM provider related errors."""


@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderName = "auto"
    model: str = "gpt-5-mini"
    api_key_env: str = "OPENAI_API_KEY"


class BaseLLM:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class MockLLM(BaseLLM):
    """
    A deterministic fake LLM for development & tests.
    No network, no keys, always available.
    """

    def generate(self, prompt: str) -> str:
        prompt_preview = prompt.strip().replace("\n", " ")[:240]
        return (
            "[mock] I can't call a real model yet, but your pipeline works.\n"
            f"[mock] Prompt preview: {prompt_preview}\n"
            "[mock] Next: set OPENAI_API_KEY (optional) to use a real model."
        )


class OpenAIResponsesLLM(BaseLLM):
    """
    OpenAI provider using the official Python SDK + Responses API.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise LLMError(
                "OpenAI SDK not installed. Install with: pip install -e \".[dev,openai]\""
            ) from e

        self._OpenAI = OpenAI
        self._model = model
        self._api_key = api_key
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            # If api_key is None, SDK will read OPENAI_API_KEY from env automatically
            self._client = self._OpenAI(api_key=self._api_key) if self._api_key else self._OpenAI()
        return self._client

    def generate(self, prompt: str) -> str:
        client = self._client_lazy()
        try:
            resp = client.responses.create(
                model=self._model,
                input=prompt,
            )
        except Exception as e:
            raise LLMError(f"OpenAI call failed: {e}") from e

        # Official examples use resp.output_text
        text = getattr(resp, "output_text", None)
        return text if isinstance(text, str) and text.strip() else str(resp)


class GeminiLLM(BaseLLM):
    """
    Gemini provider using the Google GenAI SDK.
    """

    def __init__(self, model: str, api_key: Optional[str] = None):
        try:
            from google import genai  # type: ignore
        except Exception as e:
            raise LLMError(
                "Gemini SDK not installed. Install with: pip install -e \".[dev,gemini]\""
            ) from e

        self._genai = genai
        self._model = model
        self._api_key = api_key

        if not self._api_key:
            raise LLMError("GEMINI_API_KEY not set.")

        self._client = self._genai.Client(api_key=self._api_key)

    def generate(self, prompt: str) -> str:
        try:
            resp = self._client.models.generate_content(model=self._model, contents=prompt)
        except Exception as e:
            raise LLMError(f"Gemini call failed: {e}") from e

        text = getattr(resp, "text", None)
        return text if isinstance(text, str) and text.strip() else str(resp)


def build_llm(cfg: LLMConfig) -> BaseLLM:
    """
    provider=auto:
      - If OPENAI_API_KEY is set, try OpenAI provider
      - Else fallback to Mock
    """
    openai_api_key = os.getenv(cfg.api_key_env)
    gemini_api_key = os.getenv("GEMINI_API_KEY")

    if cfg.provider == "mock":
        return MockLLM()

    if cfg.provider == "openai":
        return OpenAIResponsesLLM(model=cfg.model, api_key=openai_api_key)

    if cfg.provider == "gemini":
        model = cfg.model
        if model == "gpt-5-mini" and not os.getenv("GEMINI_MODEL"):
            model = "gemini-2.0-flash"
        return GeminiLLM(model=model, api_key=gemini_api_key)

    # auto
    if openai_api_key:
        try:
            return OpenAIResponsesLLM(model=cfg.model, api_key=openai_api_key)
        except LLMError:
            return MockLLM()

    if gemini_api_key:
        try:
            model = cfg.model
            if model == "gpt-5-mini" and not os.getenv("GEMINI_MODEL"):
                model = "gemini-2.0-flash"
            return GeminiLLM(model=model, api_key=gemini_api_key)
        except LLMError:
            return MockLLM()

    return MockLLM()
