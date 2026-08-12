"""Thin wrapper around Gemini generation, isolated behind an interface so
the backend and eval harness can both use it (and so it can be mocked in
tests without touching the network).
"""

from __future__ import annotations

from collections.abc import Iterator

from google import genai

from ragcore.config import Settings


class GenerationClient:
    def __init__(self, settings: Settings):
        genai.configure(api_key=settings.gemini_api_key)
        self._model = genai.GenerativeModel(settings.gemini_generation_model)

    def stream(self, prompt: str) -> Iterator[str]:
        response = self._model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "max_output_tokens": 1024},
            stream=True,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    def generate(self, prompt: str) -> str:
        response = self._model.generate_content(
            prompt, generation_config={"temperature": 0.1, "max_output_tokens": 1024}
        )
        return response.text or ""
