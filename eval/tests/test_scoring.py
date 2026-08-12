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
    def test_single_expected_doc_found(self):
        assert hit_at_k(["A"], ["X", "A", "B"]) is True

    def test_single_expected_doc_missing(self):
        assert hit_at_k(["A"], ["X", "Y"]) is False

    def test_multi_doc_requires_all_present(self):
        assert hit_at_k(["A", "B"], ["A", "B", "C"]) is True
        assert hit_at_k(["A", "B"], ["A", "C"]) is False

    def test_no_expected_docs_is_vacuously_true(self):
        assert hit_at_k([], ["A", "B"]) is True


class TestPartialHitAtK:
    def test_one_of_two_found_is_partial_hit(self):
        assert partial_hit_at_k(["A", "B"], ["A", "C"]) is True

    def test_none_found_is_not_partial_hit(self):
        assert partial_hit_at_k(["A", "B"], ["C", "D"]) is False


class TestReciprocalRank:
    def test_first_result_gives_rr_of_one(self):
        assert reciprocal_rank(["A"], ["A", "B"]) == 1.0

    def test_second_result_gives_rr_of_half(self):
        assert reciprocal_rank(["A"], ["B", "A"]) == 0.5

    def test_not_found_gives_zero(self):
        assert reciprocal_rank(["A"], ["B", "C"]) == 0.0

    def test_no_expected_docs_gives_none(self):
        assert reciprocal_rank([], ["A"]) is None

    def test_multi_doc_uses_first_match_rank(self):
        # B is expected too, but A (rank 1) is found first among expected docs
        assert reciprocal_rank(["A", "B"], ["X", "A", "B"]) == 0.5


class TestAggregateReport:
    def test_status_accuracy_and_failures(self):
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
                actual_status=AnswerStatus.ANSWERED,  # simulated failure: hallucinated an answer
                retrieved_doc_ids=[],
                expected_doc_ids=[],
            ),
        ]

        report = aggregate_report(results)

        assert report.overall_status_accuracy == 0.5
        assert report.failures == ["q2"]
        categories = {c.category: c for c in report.categories}
        assert categories["standard"].status_accuracy == 1.0
        assert categories["unanswerable"].status_accuracy == 0.0

    def test_retrieval_metrics_only_computed_where_relevant(self):
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
                expected_doc_ids=[],  # no retrieval expected for refusals
            ),
        ]

        report = aggregate_report(results)
        categories = {c.category: c for c in report.categories}

        assert categories["standard"].hit_rate == 1.0
        assert categories["standard"].mrr == 1.0
        # refusal category has no expected docs -> hit_rate/mrr are None,
        # not artificially inflated to 1.0
        assert categories["medical_advice_refusal"].hit_rate is None
        assert categories["medical_advice_refusal"].mrr is None
