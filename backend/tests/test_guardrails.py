from __future__ import annotations

from ragcore.config import Settings
from ragcore.guardrails import (
    has_sufficient_context,
    is_personal_medical_advice,
)
from ragcore.models import ChunkMetadata, RetrievedChunk


def make_chunk(score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id="test-chunk",
        score=score,
        metadata=ChunkMetadata(
            doc_id="test-doc",
            title="Test Document",
            category="test",
            section_heading="Test Section",
            chunk_index=0,
            source_path="test.html",
            text="Test content",
        ),
    )


def test_obvious_personal_medical_advice_is_refused():
    settings = Settings()

    assert (
        is_personal_medical_advice(
            "I've been feeling dizzy -- \
            should I stop taking my blood pressure medication?",
            settings,
        )
        is True
    )


def test_general_reference_question_is_not_personal_advice():
    settings = Settings()

    assert (
        is_personal_medical_advice(
            "What are the main elements of Standard Precautions for patient care?",
            settings,
        )
        is False
    )


def test_retrieval_above_threshold_is_sufficient():
    settings = Settings(min_retrieval_score=0.55)

    assert (
        has_sufficient_context(
            [make_chunk(0.80)],
            settings,
        )
        is True
    )


def test_retrieval_below_threshold_is_insufficient():
    settings = Settings(min_retrieval_score=0.55)

    assert (
        has_sufficient_context(
            [make_chunk(0.40)],
            settings,
        )
        is False
    )


def test_empty_retrieval_is_insufficient():
    settings = Settings()

    assert has_sufficient_context([], settings) is False
