from __future__ import annotations

import logging
import time

from google import genai
from google.genai import types

from ragcore.config import Settings

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_DELAY_SECONDS = 30
# set slightly higher to account for google input token limit of 2048 e.g 20 * 700
# (max_token_count)
_BATCH_SIZE = 20


class EmbeddingClient:
    def __init__(self, settings: Settings):
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embedding_model
        self._dimension = settings.embedding_dimension

    def embed_document(self, text: str) -> list[float]:
        return self._embed(text, task_type="RETRIEVAL_DOCUMENT")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed multiple document chunks in batches.

        more efficient than calling embed_document() every time for each chunk.
        """
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]

            embeddings = self._embed_batch(
                batch,
                task_type="RETRIEVAL_DOCUMENT",
            )

            all_embeddings.extend(embeddings)

        return all_embeddings

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

            except Exception as exc:
                last_err = exc

                logger.warning(
                    "embedding attempt %s/%s failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )

                if attempt < _MAX_RETRIES:
                    logger.info(
                        "Waiting %s seconds before retrying...",
                        _RETRY_DELAY_SECONDS,
                    )
                    time.sleep(_RETRY_DELAY_SECONDS)

        assert last_err is not None
        raise last_err

    def _embed_batch(
        self,
        texts: list[str],
        task_type: str,
    ) -> list[list[float]]:
        """
        Embed a batch of texts in a single Gemini API request.
        """

        if not texts:
            return []

        last_err: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                result = self._client.models.embed_content(
                    model=self._model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._dimension,
                    ),
                )

                if not result.embeddings:
                    raise ValueError("Gemini returned no embeddings")

                if len(result.embeddings) != len(texts):
                    raise ValueError(
                        f"Expected {len(texts)} embeddings, "
                        f"got {len(result.embeddings)}"
                    )

                embeddings: list[list[float]] = []

                for embedding in result.embeddings:
                    values = embedding.values

                    if values is None:
                        raise ValueError("Gemini returned an embedding with no values")

                    if len(values) != self._dimension:
                        raise ValueError(
                            f"Expected {self._dimension}-dimensional embedding, "
                            f"got {len(values)} dimensions"
                        )

                    embeddings.append(values)

                return embeddings

            except Exception as exc:
                last_err = exc

                logger.warning(
                    "batch embedding attempt %s/%s failed: %s",
                    attempt,
                    _MAX_RETRIES,
                    exc,
                )

                if attempt < _MAX_RETRIES:
                    logger.info(
                        "Waiting %s seconds before retrying...",
                        _RETRY_DELAY_SECONDS,
                    )
                    time.sleep(_RETRY_DELAY_SECONDS)

        assert last_err is not None
        raise last_err
