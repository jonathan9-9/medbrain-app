"""LLM-as-judge scoring for the answerable-question categories. Used as a
secondary signal on top of the deterministic status check in scoring.py --
groundedness/correctness are graded 1-5 by Gemini itself, which is a
defensible-but-imperfect approach documented as a limitation in
DESIGN.md's failure analysis (a judge sharing a model family with the
generator can share blind spots).
"""

from __future__ import annotations

import json
import logging

from ragcore.config import Settings
from ragcore.llm import GenerationClient
from ragcore.models import RetrievedChunk

logger = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are grading an AI assistant's answer for a clinical \
operations document-lookup tool. Score two dimensions from 1 (worst) to 5 \
(best):

- groundedness: does every factual claim in the ANSWER trace back to the \
RETRIEVED CONTEXT below? A 5 means no claim goes beyond the context. A 1 \
means the answer contains claims unsupported by the context.
- correctness: does the ANSWER match the substance of the EXPECTED SUMMARY \
(written by a human reviewer)? A 5 means it covers the same key facts. A 1 \
means it misses or contradicts the expected summary.

QUESTION: {question}

RETRIEVED CONTEXT:
{context}

EXPECTED SUMMARY (human-written, for reference only -- the answer doesn't \
need to match its wording):
{expected_summary}

ANSWER TO GRADE:
{answer}

Respond with strict JSON only, no markdown fences:
{{"groundedness": <1-5>, "correctness": <1-5>, "justification": "<one sentence>"}}
"""


def judge_answer(
    settings: Settings,
    generator: GenerationClient,
    question: str,
    answer: str,
    expected_summary: str,
    retrieved: list[RetrievedChunk],
) -> tuple[int | None, int | None, str]:
    context = "\n\n".join(
        f"[{c.metadata.title} - {c.metadata.section_heading}] {c.metadata.text}"
        for c in retrieved
    )
    prompt = _JUDGE_PROMPT.format(
        question=question,
        context=context,
        expected_summary=expected_summary,
        answer=answer,
    )
    try:
        raw = generator.generate(prompt)
        cleaned = (
            raw.strip()
            .removeprefix("```json")
            .removeprefix("```")
            .removesuffix("```")
            .strip()
        )
        parsed = json.loads(cleaned)
        return (
            int(parsed["groundedness"]),
            int(parsed["correctness"]),
            str(parsed.get("justification", "")),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge call failed for question %r: %s", question, exc)
        return None, None, f"judge_error: {exc}"
