from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types

from ragcore.config import Settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BASE_DELAY_SECONDS = 1.5


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model
        self._dimension = settings.embedding_dimension

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, task_type="RETRIEVAL_QUERY")

    def _embed(self, text: str, task_type: str) -> list[float]:
        last_err: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._dimension,
                    ),
                )

                if not result.embeddings:
                    raise ValueError("Gemini returned no embeddings")

                values = result.embeddings[0].values

                if values is None:
                    raise ValueError("Gemini returned an embedding with no values")

                if len(values) != self._dimension:
                    raise ValueError(
                        f"Expected {self._dimension}-dimensional embedding, "
                        f"got {len(values)} dimensions"
                    )

                return values

            except Exception as exc:  # noqa: BLE001
                last_err = exc

                logger.warning(
                    "embedding attempt %s/%s failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )

                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BASE_DELAY_SECONDS * attempt)

        assert last_err is not None
        raise last_err
