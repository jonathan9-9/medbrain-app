"""Loads eval/dataset.yaml into typed EvalCase objects."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel


class EvalCategory(str, Enum):
    STANDARD = "standard"
    MULTI_DOC_SYNTHESIS = "multi_doc_synthesis"
    UNANSWERABLE = "unanswerable"
    MEDICAL_ADVICE_REFUSAL = "medical_advice_refusal"


class EvalCase(BaseModel):
    id: str
    category: EvalCategory
    question: str
    expected_source_doc_ids: list[str] = []
    expected_answer_summary: str | None = None


def load_dataset(path: Path) -> list[EvalCase]:
    raw = yaml.safe_load(path.read_text())
    return [EvalCase(**item) for item in raw]
