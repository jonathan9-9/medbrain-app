from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.ragcore.models import AnswerStatus  # noqa: E402
from eval.dataset import EvalCategory  # noqa: E402
from eval.scoring import (  # noqa: E402
    CaseResult,
    aggregate_report,
    hit_at_k,
    partial_hit_at_k,
    reciprocal_rank,
)


class TestHitAtK:
    def test_expected_document_found(self):
        assert hit_at_k(["A"], ["X", "A", "B"]) is True

    def test_expected_document_missing(self):
        assert hit_at_k(["A"], ["X", "Y"]) is False

    def test_multi_document_requires_all_sources(self):
        assert hit_at_k(["A", "B"], ["A", "B", "C"]) is True
        assert hit_at_k(["A", "B"], ["A", "C"]) is False

    def test_no_expected_documents_is_vacuously_true(self):
        assert hit_at_k([], ["A", "B"]) is True


class TestPartialHitAtK:
    def test_one_expected_source_is_partial_hit(self):
        assert partial_hit_at_k(["A", "B"], ["A", "C"]) is True

    def test_no_expected_source_is_not_partial_hit(self):
        assert partial_hit_at_k(["A", "B"], ["C", "D"]) is False


class TestReciprocalRank:
    def test_first_result_has_rr_one(self):
        assert reciprocal_rank(["A"], ["A", "B"]) == 1.0

    def test_second_result_has_rr_one_half(self):
        assert reciprocal_rank(["A"], ["B", "A"]) == 0.5

    def test_missing_expected_source_has_rr_zero(self):
        assert reciprocal_rank(["A"], ["B", "C"]) == 0.0

    def test_no_expected_sources_have_no_rr(self):
        assert reciprocal_rank([], ["A"]) is None

    def test_multi_document_rr_uses_first_expected_source(self):
        assert reciprocal_rank(["A", "B"], ["X", "A", "B"]) == 0.5


class TestAggregateReport:
    def test_status_failures_are_reported(self):
        results = [
            CaseResult(
                case_id="q1",
                category=EvalCategory.STANDARD,
                expected_status=AnswerStatus.ANSWERED,
                actual_status=AnswerStatus.ANSWERED,
                retrieved_doc_ids=["A"],
                expected_doc_ids=["A"],
            ),
            CaseResult(
                case_id="q2",
                category=EvalCategory.UNANSWERABLE,
                expected_status=AnswerStatus.UNANSWERABLE,
                actual_status=AnswerStatus.ANSWERED,
                retrieved_doc_ids=[],
                expected_doc_ids=[],
            ),
        ]

        report = aggregate_report(results)

        assert report.overall_status_accuracy == 0.5
        assert report.failures == ["q2"]
        assert report.errors == []
        assert report.evaluated_cases == 2

    def test_evaluation_errors_do_not_reduce_status_accuracy(self):
        results = [
            CaseResult(
                case_id="q1",
                category=EvalCategory.STANDARD,
                expected_status=AnswerStatus.ANSWERED,
                actual_status=AnswerStatus.ANSWERED,
                retrieved_doc_ids=["A"],
                expected_doc_ids=["A"],
            ),
            CaseResult(
                case_id="q2",
                category=EvalCategory.STANDARD,
                expected_status=AnswerStatus.ANSWERED,
                actual_status=None,
                retrieved_doc_ids=[],
                expected_doc_ids=["A"],
                error="429 RESOURCE_EXHAUSTED",
            ),
        ]

        report = aggregate_report(results)

        assert report.total_cases == 2
        assert report.evaluated_cases == 1
        assert report.error_count == 1
        assert report.overall_status_accuracy == 1.0
        assert report.failures == []
        assert report.errors == ["q2"]

        standard = {category.category: category for category in report.categories}[
            "standard"
        ]

        assert standard.n == 2
        assert standard.evaluated_n == 1
        assert standard.error_count == 1
        assert standard.status_accuracy == 1.0

    def test_retrieval_metrics_only_apply_to_answerable_cases(self):
        results = [
            CaseResult(
                case_id="q1",
                category=EvalCategory.STANDARD,
                expected_status=AnswerStatus.ANSWERED,
                actual_status=AnswerStatus.ANSWERED,
                retrieved_doc_ids=["A", "B"],
                expected_doc_ids=["A"],
            ),
            CaseResult(
                case_id="q2",
                category=EvalCategory.MEDICAL_ADVICE_REFUSAL,
                expected_status=AnswerStatus.REFUSED_MEDICAL_ADVICE,
                actual_status=AnswerStatus.REFUSED_MEDICAL_ADVICE,
                retrieved_doc_ids=[],
                expected_doc_ids=[],
            ),
        ]

        report = aggregate_report(results)

        categories = {category.category: category for category in report.categories}

        assert categories["standard"].hit_rate == 1.0
        assert categories["standard"].mrr == 1.0

        assert categories["medical_advice_refusal"].hit_rate is None
        assert categories["medical_advice_refusal"].mrr is None
