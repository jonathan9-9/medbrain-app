"""Two independent guardrails that run BEFORE generation, as hard gates
rather than prompt-only instructions (see DESIGN.md section 4):

1. Personal-medical-advice detection: a fast regex pre-filter for
   clear-cut personal medical advice, followed by an LLM classifier only
   when a question contains signals that it may be individualized medical
   advice. This avoids spending an LLM call on ordinary clinical-reference
   questions.

2. Retrieval sufficiency check: if the best retrieved chunk's similarity
   score is below the configured threshold, we do not generate at all —
   we return a fixed "not covered by this corpus" response. This makes the
   "unanswerable" behavior deterministic rather than hoping the model
   chooses to hedge.
"""

from __future__ import annotations

import logging
import re

from google import genai
from google.genai import types
from pydantic import BaseModel

from ragcore.config import Settings
from ragcore.models import RetrievedChunk

logger = logging.getLogger(__name__)


# High-confidence patterns that can immediately identify personal medical
# advice without an LLM call.
_PERSONAL_MEDICAL_ADVICE_PATTERNS = [
    r"\bshould i\s+(stop|start|take|skip|increase|decrease|lower|raise|double|halve)\b.{0,40}\b(my|this)\b",  # noqa: E501
    r"\bis it (safe|ok|okay) for me to\b",
    r"\bcan i (stop|start|take|skip)\b.{0,40}\bmy\b",
    r"\bwhat dose should i\b",
    r"\bwhat (medication|medicine|drug) and dose should i give\b",
    r"\bdo i need to see a doctor\b.{0,30}\bmy\b",
    r"\bam i having a (heart attack|stroke|allergic reaction)\b",
]


# Signals used to decide whether the more expensive LLM classifier is
# actually necessary. Ordinary clinical-reference questions such as
# "What temperature should vaccines be stored at?" do not have these
# personal-context signals and therefore skip the classifier entirely.
_PERSONAL_CONTEXT_PATTERN = re.compile(
    r"\b(i|i'm|i've|i'd|i'll|me|my|mine|myself)\b"
    r"|\b(this|the specific|the named)\s+patient\b",
    re.IGNORECASE,
)

_MEDICAL_DECISION_PATTERN = re.compile(
    r"\b("
    r"should i|"
    r"what should i|"
    r"what do i|"
    r"can i|"
    r"do i need|"
    r"is it safe|"
    r"dose|"
    r"dosage|"
    r"medication|"
    r"medicine|"
    r"drug|"
    r"treatment|"
    r"symptom|"
    r"fever|"
    r"dizzy|"
    r"headache|"
    r"pain|"
    r"stop taking|"
    r"start taking"
    r")\b",
    re.IGNORECASE,
)


_CLASSIFIER_PROMPT = """You are a safety classifier for a clinical-operations
document lookup tool used by healthcare professionals.

Determine whether the question asks for PERSONAL medical advice about an
individual's own or a specific patient's diagnosis, treatment, medication,
medication dose, or symptoms.

Classify as true when the question asks what should happen for that
individual or specific patient.

Classify as false for:
- general medical reference questions
- clinical guidelines
- policies and procedures
- organizational or operational questions
- general questions about what healthcare staff should do

Question:
{question}
"""


class MedicalAdviceClassification(BaseModel):
    """Structured result returned by the Gemini safety classifier."""

    is_personal_medical_advice: bool


def is_personal_medical_advice(question: str, settings: Settings) -> bool:
    lowered = question.lower()

    # Gate 1: obvious personal medical advice.
    for pattern in _PERSONAL_MEDICAL_ADVICE_PATTERNS:
        if re.search(pattern, lowered):
            logger.debug("medical-advice guardrail matched regex")
            return True

    # Gate 2: only spend a Gemini call when the question actually contains
    # both personal-context and medical-decision signals.
    if not _looks_like_personal_medical_advice_candidate(lowered):
        return False

    return _classify_with_llm(question, settings)


def _looks_like_personal_medical_advice_candidate(question: str) -> bool:
    return bool(
        _PERSONAL_CONTEXT_PATTERN.search(question)
        and _MEDICAL_DECISION_PATTERN.search(question)
    )


def _classify_with_llm(question: str, settings: Settings) -> bool:
    raw_text = ""

    try:
        client = genai.Client(api_key=settings.gemini_api_key)

        response = client.models.generate_content(
            model=settings.gemini_generation_model,
            contents=_CLASSIFIER_PROMPT.format(question=question),
            config=types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=64,
                response_mime_type="application/json",
                response_schema=MedicalAdviceClassification,
            ),
        )

        raw_text = (response.text or "").strip()

        if not raw_text:
            raise ValueError("classifier returned an empty response")

        parsed = MedicalAdviceClassification.model_validate_json(raw_text)

        return parsed.is_personal_medical_advice

    except Exception:  # noqa: BLE001
        # If the classifier is unavailable or returns an unusable response,
        # allow the request to continue. The regex pass above catches the
        # highest-confidence unsafe cases.
        logger.exception(
            "medical-advice classifier call failed; raw response was: %r",
            raw_text,
        )
        return False


def has_sufficient_context(
    retrieved: list[RetrievedChunk],
    settings: Settings,
) -> bool:
    if not retrieved:
        return False

    return retrieved[0].score >= settings.min_retrieval_score
