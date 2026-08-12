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

import google.generativeai as genai

from ragcore.config import Settings
from ragcore.models import RetrievedChunk

logger = logging.getLogger(__name__)

# Deliberately narrow, high-precision patterns for the obvious cases.
# Anything not caught here falls through to the LLM classifier below,
# rather than being assumed safe.
_PERSONAL_MEDICAL_ADVICE_PATTERNS = [
    (
        r"\bshould i\s+"
        r"(stop|start|take|skip|increase|decrease|lower|raise|double|halve)"
        r"\b.{0,40}\b(my|this)\b"
    ),
    r"\bis it (safe|ok|okay) for me to\b",
    r"\bcan i (stop|start|take|skip)\b.{0,40}\bmy\b",
    r"\bwhat dose should i\b",
    r"\bdo i need to see a doctor\b.{0,30}\bmy\b",
    r"\bam i having a (heart attack|stroke|allergic reaction)\b",
]

_CLASSIFIER_PROMPT = """You are a safety classifier for a clinical-operations \
document lookup tool used by healthcare professionals (not patients). \
Classify whether the following question is asking for PERSONAL medical \
advice about an individual's own (or a specific named patient's) diagnosis, \
treatment, medication dosing, or symptoms -- as opposed to asking about \
organizational policy, a procedure, or general reference information.

Question: {question}

Respond with strict JSON only, no markdown: {{"is_personal_medical_advice": true|false}}
"""


def is_personal_medical_advice(question: str, settings: Settings) -> bool:
    lowered = question.lower()
    for pattern in _PERSONAL_MEDICAL_ADVICE_PATTERNS:
        if re.search(pattern, lowered):
            return True
    return _classify_with_llm(question, settings)


def _classify_with_llm(question: str, settings: Settings) -> bool:
    try:
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_generation_model)
        response = model.generate_content(
            _CLASSIFIER_PROMPT.format(question=question),
            generation_config={"temperature": 0, "max_output_tokens": 50},
        )
        text = (response.text or "").strip()
        text = (
            text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        )
        parsed = json.loads(text)
        return bool(parsed.get("is_personal_medical_advice", False))
    except Exception:
        # Fail closed-ish: if the classifier itself errors, we do NOT
        # silently treat the question as safe. We log and let the
        # sufficiency/generation path continue, since the regex pass
        # already covers the highest-confidence unsafe cases -- but we
        # surface the failure so it's visible in logs/monitoring.
        logger.exception("medical-advice classifier call failed")
        return False


def has_sufficient_context(retrieved: list[RetrievedChunk], settings: Settings) -> bool:
    if not retrieved:
        return False
    return retrieved[0].score >= settings.min_retrieval_score
