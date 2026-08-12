"""Prompt construction. The citation contract described here is enforced
by the backend afterward (see backend/app/services/citations.py) -- the
model choosing to tag [S2] only means "point at retrieved chunk 2"; the
backend, not the model, decides what doc/section that tag actually links
to.
"""

from __future__ import annotations

from ragcore.models import RetrievedChunk

SYSTEM_INSTRUCTIONS = """You are a health-information document-lookup \
assistant. You help users find and understand information in a hand-curated \
collection of authoritative health guidance and medication-label documents.

Rules you must follow exactly:
1. Answer ONLY using the numbered source excerpts provided below. Do not \
use any outside knowledge, even if you believe it to be correct.
2. Every factual claim in your answer must end with the tag(s) of the \
source(s) it came from, like this: "Hand hygiene must be performed before \
touching a patient [S1]." If a sentence draws on two sources, tag both: \
"...[S1][S3]".
3. If the provided sources do not fully answer the question, say plainly \
what is and is not covered rather than filling gaps with outside knowledge.
4. You are not a source of personal medical advice. If asked to advise on \
an individual's diagnosis, treatment, or medication decisions, do not \
answer the clinical question -- state that this tool is for document \
lookup and the question should go to a licensed clinician.
5. Write in clear, concise language. Explain the documents' information, \
but do not turn it into individualized medical advice.
"""


def build_generation_prompt(question: str, retrieved: list[RetrievedChunk]) -> str:
    sources_block = "\n\n".join(
        f"[S{i + 1}] Document: {chunk.metadata.title} "
        f"(Section: {chunk.metadata.section_heading})\n{chunk.metadata.text}"
        for i, chunk in enumerate(retrieved)
    )
    return (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"--- SOURCE EXCERPTS ---\n{sources_block}\n--- END SOURCE EXCERPTS ---\n\n"
        f"Question: {question}\n\nAnswer (remember: tag every claim, e.g. [S1]):"
    )


UNANSWERABLE_TEMPLATE = (
    "I couldn't find information about that in the indexed document corpus. "
    "This tool only answers from the hand-curated health documents that have "
    "been ingested -- it doesn't have outside knowledge "
    "to fall back on. You may want to check with the relevant department "
    "directly, or ask about a related topic that IS covered, such as: {topics}."
)

MEDICAL_ADVICE_REFUSAL = (
    "I'm a health-information document-lookup tool, not a source "
    "of personal medical advice. I can't advise on an individual's "
    "diagnosis, treatment, or medication decisions. If this is about your "
    "own care, please contact your clinician or care team. If you're "
    "looking for the organization's policy or procedure on this topic "
    "(rather than advice for a specific person), feel free to rephrase the "
    "question that way and I'm glad to help."
)
