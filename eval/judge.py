"""LLM-as-judge scoring for selected answerable evaluation cases.

Groundedness and correctness are graded from 1-5 by Gemini. Deterministic
status and retrieval metrics are still computed for every eval case; this
judge is an additional answer-quality signal on a small, representative
subset so the harness remains practical under API quotas.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from backend.ragcore.llm import GenerationClient
from backend.ragcore.models import RetrievedChunk

logger = logging.getLogger(__name__)


_JUDGE_PROMPT = """You are grading an AI assistant's answer for a clinical
operations document-lookup tool.

Score these two dimensions from 1 (worst) to 5 (best):

1. groundedness:
   Does every factual claim in the ANSWER trace back to the RETRIEVED
   CONTEXT? A 5 means the answer contains no factual claims unsupported
   by the retrieved context. A 1 means many claims are unsupported.

2. correctness:
   Does the ANSWER match the substance of the EXPECTED SUMMARY written
   by a human reviewer? A 5 means it covers the same key facts without
   contradiction. A 1 means it misses or contradicts the expected summary.

Do not grade writing style. Grade factual grounding and substantive
correctness.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

EXPECTED SUMMARY:
{expected_summary}

ANSWER TO GRADE:
{answer}
"""


class JudgeResult(BaseModel):
    groundedness: int = Field(ge=1, le=5)
    correctness: int = Field(ge=1, le=5)
    justification: str = ""


def judge_answer(
    generator: GenerationClient,
    question: str,
    answer: str,
    expected_summary: str,
    retrieved: list[RetrievedChunk],
) -> tuple[int | None, int | None, str]:
    context = "\n\n".join(
        f"[{chunk.metadata.title} - {chunk.metadata.section_heading}] "
        f"{chunk.metadata.text}"
        for chunk in retrieved
    )

    prompt = _JUDGE_PROMPT.format(
        question=question,
        context=context,
        expected_summary=expected_summary,
        answer=answer,
    )

    try:
        raw = generator.generate_structured(
            prompt=prompt,
            response_schema=JudgeResult,
        ).strip()

        if not raw:
            raise ValueError("judge returned an empty response")

        parsed = JudgeResult.model_validate_json(raw)

        return (
            parsed.groundedness,
            parsed.correctness,
            parsed.justification,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "judge call failed for question %r: %s",
            question,
            exc,
        )
        return None, None, f"judge_error: {exc}"
