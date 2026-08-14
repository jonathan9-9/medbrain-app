"""Pure, network-free scoring functions.

Kept separate from run_eval.py's orchestration so the actual metric math
(hit rate, MRR, status mapping, aggregation) can be unit-tested without
hitting Gemini or Pinecone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from backend.ragcore.models import AnswerStatus
from eval.dataset import EvalCategory

# Which AnswerStatus a correctly-behaving system should produce for each
# category. This is the most safety-relevant check in the harness because
# it verifies that the application answers, refuses, or declines as expected.
EXPECTED_STATUS_BY_CATEGORY: dict[EvalCategory, AnswerStatus] = {
    EvalCategory.STANDARD: AnswerStatus.ANSWERED,
    EvalCategory.MULTI_DOC_SYNTHESIS: AnswerStatus.ANSWERED,
    EvalCategory.UNANSWERABLE: AnswerStatus.UNANSWERABLE,
    EvalCategory.MEDICAL_ADVICE_REFUSAL: AnswerStatus.REFUSED_MEDICAL_ADVICE,
}


def hit_at_k(
    expected_doc_ids: list[str],
    retrieved_doc_ids: list[str],
) -> bool:
    """True if every expected doc_id appears in the retrieved set.

    For single-document cases this behaves like normal hit@k.
    For multi-document cases all expected documents must be present.
    """

    if not expected_doc_ids:
        return True

    retrieved_set = set(retrieved_doc_ids)

    return all(doc_id in retrieved_set for doc_id in expected_doc_ids)


def partial_hit_at_k(
    expected_doc_ids: list[str],
    retrieved_doc_ids: list[str],
) -> bool:
    """True if at least one expected document was retrieved."""

    if not expected_doc_ids:
        return True

    retrieved_set = set(retrieved_doc_ids)

    return any(doc_id in retrieved_set for doc_id in expected_doc_ids)


def reciprocal_rank(
    expected_doc_ids: list[str],
    retrieved_doc_ids: list[str],
) -> float | None:
    """Return 1/rank of the first expected source found.

    None is returned when there are no expected sources, such as an
    unanswerable or medical-advice-refusal case.
    """

    if not expected_doc_ids:
        return None

    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return 1.0 / rank

    return 0.0


@dataclass
class CaseResult:
    case_id: str
    category: EvalCategory
    expected_status: AnswerStatus
    actual_status: AnswerStatus | None
    retrieved_doc_ids: list[str] = field(default_factory=list)
    expected_doc_ids: list[str] = field(default_factory=list)
    has_citations: bool = False
    groundedness_score: int | None = None
    correctness_score: int | None = None
    judge_notes: str | None = None
    error: str | None = None

    @property
    def evaluation_error(self) -> bool:
        """True when the case could not be evaluated because execution failed."""

        return self.error is not None

    @property
    def status_correct(self) -> bool:
        """Only successful evaluations can be counted as status correct."""

        if self.evaluation_error or self.actual_status is None:
            return False

        return self.actual_status == self.expected_status

    @property
    def hit(self) -> bool:
        return hit_at_k(
            self.expected_doc_ids,
            self.retrieved_doc_ids,
        )

    @property
    def partial_hit(self) -> bool:
        return partial_hit_at_k(
            self.expected_doc_ids,
            self.retrieved_doc_ids,
        )

    @property
    def rr(self) -> float | None:
        return reciprocal_rank(
            self.expected_doc_ids,
            self.retrieved_doc_ids,
        )


class CategoryReport(BaseModel):
    category: str
    n: int
    evaluated_n: int
    error_count: int
    status_accuracy: float | None = None
    hit_rate: float | None = None
    partial_hit_rate: float | None = None
    mrr: float | None = None
    avg_groundedness: float | None = None
    avg_correctness: float | None = None


class EvalReport(BaseModel):
    total_cases: int
    evaluated_cases: int
    error_count: int
    overall_status_accuracy: float | None
    overall_mrr: float | None
    judged_case_count: int
    judged_case_ids: list[str]
    categories: list[CategoryReport]
    failures: list[str]
    errors: list[str]


def aggregate_report(results: list[CaseResult]) -> EvalReport:
    by_category: dict[EvalCategory, list[CaseResult]] = {}

    for result in results:
        by_category.setdefault(
            result.category,
            [],
        ).append(result)

    category_reports: list[CategoryReport] = []

    for category, cases in by_category.items():
        n = len(cases)

        successful_cases = [
            case
            for case in cases
            if not case.evaluation_error and case.actual_status is not None
        ]

        evaluated_n = len(successful_cases)

        error_count = sum(case.evaluation_error for case in cases)

        status_acc = (
            sum(case.actual_status == case.expected_status for case in successful_cases)
            / evaluated_n
            if evaluated_n
            else None
        )

        rr_values = [
            case.rr
            for case in cases
            if case.rr is not None and not case.evaluation_error
        ]

        hits = [
            case.hit
            for case in cases
            if case.expected_doc_ids and not case.evaluation_error
        ]

        partial_hits = [
            case.partial_hit
            for case in cases
            if case.expected_doc_ids and not case.evaluation_error
        ]

        grounded = [
            case.groundedness_score
            for case in cases
            if case.groundedness_score is not None
        ]

        correct = [
            case.correctness_score
            for case in cases
            if case.correctness_score is not None
        ]

        category_reports.append(
            CategoryReport(
                category=category.value,
                n=n,
                evaluated_n=evaluated_n,
                error_count=error_count,
                status_accuracy=(
                    round(status_acc, 3) if status_acc is not None else None
                ),
                hit_rate=(round(sum(hits) / len(hits), 3) if hits else None),
                partial_hit_rate=(
                    round(
                        sum(partial_hits) / len(partial_hits),
                        3,
                    )
                    if partial_hits
                    else None
                ),
                mrr=(round(sum(rr_values) / len(rr_values), 3) if rr_values else None),
                avg_groundedness=(
                    round(sum(grounded) / len(grounded), 2) if grounded else None
                ),
                avg_correctness=(
                    round(sum(correct) / len(correct), 2) if correct else None
                ),
            )
        )

    successful_results = [
        result
        for result in results
        if not result.evaluation_error and result.actual_status is not None
    ]

    all_rr = [
        result.rr
        for result in results
        if result.rr is not None and not result.evaluation_error
    ]

    judged_case_ids = [
        result.case_id
        for result in results
        if (
            result.groundedness_score is not None
            or result.correctness_score is not None
        )
    ]

    failures = [
        result.case_id for result in successful_results if not result.status_correct
    ]

    errors = [result.case_id for result in results if result.evaluation_error]

    overall_status_accuracy = (
        sum(result.status_correct for result in successful_results)
        / len(successful_results)
        if successful_results
        else None
    )

    return EvalReport(
        total_cases=len(results),
        evaluated_cases=len(successful_results),
        error_count=len(errors),
        overall_status_accuracy=(
            round(overall_status_accuracy, 3)
            if overall_status_accuracy is not None
            else None
        ),
        overall_mrr=(round(sum(all_rr) / len(all_rr), 3) if all_rr else None),
        judged_case_count=len(judged_case_ids),
        judged_case_ids=judged_case_ids,
        categories=sorted(
            category_reports,
            key=lambda category: category.category,
        ),
        failures=failures,
        errors=errors,
    )
