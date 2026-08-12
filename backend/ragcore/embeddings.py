"""Thin wrapper around Google's embedding API (text-embedding-004, free
tier). Both the ingestion pipeline (embedding chunks) and the backend
(embedding the user's question at query time) go through this single
module, so the two can never end up using different models/params and
silently degrading retrieval quality.
"""
from __future__ import annotations

import logging
import time

import google.generativeai as genai

from ragcore.config import Settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY_SECONDS = 1.5


class EmbeddingClient:
    def __init__(self, settings: Settings):
        genai.configure(api_key=settings.gemini_api_key)
        self._model = f"models/{settings.gemini_embedding_model}"

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, task_type="retrieval_document")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, task_type="retrieval_query")

    def _embed(self, text: str, task_type: str) -> list[float]:
        last_err: Exception | None = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = genai.embed_content(
                    model=self._model,
                    content=text,
                    task_type=task_type,
                )
                return result["embedding"]
            except Exception as exc:  # noqa: BLE001 - broad by design, see retry loop
                last_err = exc
                logger.warning(
                    "embedding attempt %s/%s failed: %s", attempt, _MAX_RETRIES, exc
                )
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BASE_DELAY_SECONDS * attempt)
        assert last_err is not None
        raise last_err
