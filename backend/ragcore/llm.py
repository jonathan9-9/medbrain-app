"""Thin wrapper around Gemini generation, isolated behind an interface so
the backend and eval harness can both use it (and so it can be mocked in
tests without touching the network).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from google import genai
from google.genai import types

from ragcore.config import Settings

logger = logging.getLogger(__name__)


class GenerationClient:
    def __init__(self, settings: Settings):
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_generation_model

    def stream(self, prompt: str) -> Iterator[str]:
        response = self._client.models.generate_content_stream(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )
        for chunk in response:
            logger.debug("Gemini chunk: %r", chunk)
            if chunk.text:
                yield chunk.text

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=4096,
            ),
        )

        return response.text or ""
