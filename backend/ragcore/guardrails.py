"""Two independent guardrails that run BEFORE generation, as hard gates
rather than prompt-only instructions (see DESIGN.md section 4):

1. Personal-medical-advice detection: a fast regex pre-filter for the
   clear-cut "should I take/stop my medication" phrasing, with an LLM
   classifier fallback for ambiguous cases. This is deliberately
   conservative — a false positive (refusing a borderline-legitimate
   question) is treated as safer than a false negative in this domain.

2. Retrieval sufficiency check: if the best retrieved chunk's similarity
   score is below the configured threshold, we do not generate at all —
   we return a fixed "not covered by this corpus" response. This is what
   makes the "unanswerable" behavior deterministic instead of hoping the
   model chooses to hedge.
"""

from __future__ import annotations

import json
import logging
import re

from google import genai
from google.genai import types

from ragcore.config import Settings
from ragcore.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Deliberately narrow, high-precision patterns for the obvious cases.
# Anything not caught here falls through to the LLM classifier below,
# rather than being assumed safe.
_PERSONAL_MEDICAL_ADVICE_PATTERNS = [
    r"\bshould i\s+(stop|start|take|skip|increase|decrease|lower|raise|double|halve)\b.{0,40}\b(my|this)\b",
    r"\bis it (safe|ok|okay) for me to\b",
    r"\bcan i (stop|start|take|skip)\b.{0,40}\bmy\b",
    r"\bwhat dose should i\b",
    r"\bdo i need to see a doctor\b.{0,30}\bmy\b",
    r"\bam i having a (heart attack|stroke|allergic reaction)\b",
]

_CLASSIFIER_PROMPT = """You are a safety classifier for a clinical-operations
document lookup tool used by healthcare professionals (not patients).

Determine whether the following question asks for PERSONAL medical advice
about an individual's own or a specific named patient's diagnosis,
treatment, medication dosing, or symptoms.

Return true only for personal medical advice. Return false for general
reference questions, clinical guidelines, policies, procedures, or
organizational information.

Question: {question}
"""


def is_personal_medical_advice(question: str, settings: Settings) -> bool:
    lowered = question.lower()

    for pattern in _PERSONAL_MEDICAL_ADVICE_PATTERNS:
        if re.search(pattern, lowered):
            return True

    return _classify_with_llm(question, settings)


def _classify_with_llm(question: str, settings: Settings) -> bool:
    raw_text = ""

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        response = client.models.generate_content(
            model=settings.gemini_generation_model,
            contents=_CLASSIFIER_PROMPT.format(question=question),
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=128,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "is_personal_medical_advice": {
                            "type": "BOOLEAN",
                        },
                    },
                    "required": ["is_personal_medical_advice"],
                },
            ),
        )

        raw_text = (response.text or "").strip()

        return _parse_classifier_response(raw_text)

    except Exception:  # noqa: BLE001
        # If the classifier is unavailable or returns an unusable response,
        # allow the request to continue. The regex pass above already catches
        # the highest-confidence personal-medical-advice patterns.
        logger.exception(
            "medical-advice classifier call failed; raw response was: %r",
            raw_text,
        )
        return False


def _parse_classifier_response(text: str) -> bool:
    """Parse the structured JSON returned by the Gemini classifier."""

    parsed = json.loads(text)

    return bool(parsed["is_personal_medical_advice"])


def has_sufficient_context(
    retrieved: list[RetrievedChunk],
    settings: Settings,
) -> bool:
    if not retrieved:
        return False

    return retrieved[0].score >= settings.min_retrieval_score
