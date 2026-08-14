"""Orchestrates a single chat turn. This is the one place that wires
guardrails, retrieval, and generation together, and it's written as a
generator of typed events so both the SSE route and tests can drive it
without needing an HTTP round trip.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator

from ragcore.citations import extract_used_citations
from ragcore.config import Settings
from ragcore.embeddings import EmbeddingClient
from ragcore.guardrails import has_sufficient_context, is_personal_medical_advice
from ragcore.llm import GenerationClient
from ragcore.models import AnswerStatus, ChatMessageEvent, Citation, RetrievedChunk
from ragcore.prompts import (
    MEDICAL_ADVICE_REFUSAL,
    UNANSWERABLE_TEMPLATE,
    build_generation_prompt,
)
from ragcore.vectorstore import VectorStore

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingClient,
        store: VectorStore,
        generator: GenerationClient,
    ):
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._generator = generator

    def answer(
        self,
        question: str,
        on_retrieval: Callable[[list[RetrievedChunk]], None] | None = None,
    ) -> Iterator[ChatMessageEvent]:
        # Gate 1: personal medical advice. Runs before retrieval so we
        # never spend a retrieval/generation call on a question we're
        # going to decline anyway.
        if is_personal_medical_advice(question, self._settings):
            logger.info("guardrail: refused as personal medical advice")
            yield ChatMessageEvent(
                type="status", data=AnswerStatus.REFUSED_MEDICAL_ADVICE.value
            )
            yield ChatMessageEvent(type="token", data=MEDICAL_ADVICE_REFUSAL)
            yield ChatMessageEvent(type="citations", data=[])
            yield ChatMessageEvent(type="done", data=None)
            return

        query_embedding = self._embedder.embed_query(question)

        retrieved = self._store.query(
            query_embedding,
            top_k=self._settings.retrieval_top_k,
        )

        # The eval harness uses this callback to reuse the exact chunks
        # already retrieved for the live ChatService path. Production callers
        # simply omit the callback.
        if on_retrieval is not None:
            on_retrieval(retrieved)

        yield ChatMessageEvent(
            type="retrieval",
            data=[chunk.metadata.doc_id for chunk in retrieved],
        )

        # Gate 2: retrieval sufficiency. If the best match is weak, don't
        # let the model try anyway -- return a fixed, honest response.
        if not has_sufficient_context(retrieved, self._settings):
            logger.info("guardrail: insufficient retrieval context (top score too low)")
            topics = _sample_topics(retrieved)
            yield ChatMessageEvent(
                type="status",
                data=AnswerStatus.UNANSWERABLE.value,
            )
            yield ChatMessageEvent(
                type="token",
                data=UNANSWERABLE_TEMPLATE.format(topics=topics),
            )
            yield ChatMessageEvent(type="citations", data=[])
            yield ChatMessageEvent(type="done", data=None)
            return

        yield ChatMessageEvent(
            type="status",
            data=AnswerStatus.ANSWERED.value,
        )

        prompt = build_generation_prompt(question, retrieved)

        full_answer = ""

        for token in self._generator.stream(prompt):
            full_answer += token
            yield ChatMessageEvent(type="token", data=token)

        citations: list[Citation] = extract_used_citations(
            full_answer,
            retrieved,
        )

        yield ChatMessageEvent(
            type="citations",
            data=citations,
        )
        yield ChatMessageEvent(type="done", data=None)


def _sample_topics(retrieved: list[RetrievedChunk], limit: int = 3) -> str:
    titles: list[str] = []

    for chunk in retrieved[:limit]:
        if chunk.metadata.title not in titles:
            titles.append(chunk.metadata.title)

    return ", ".join(titles) if titles else "the topics in the indexed corpus"
