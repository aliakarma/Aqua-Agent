"""tests/test_metrics.py — Unit tests for evaluation metric functions."""

import numpy as np
import pytest
from src.evaluation.metrics import (
    compute_accuracy, compute_precision, compute_recall,
    compute_f1, compute_auc, compute_wlr, compute_pcr, compute_rt,
    compute_all_detection_metrics,
)


class TestDetectionMetrics:
    def test_perfect_classifier(self):
        labels = np.array([1, 1, 0, 0, 1, 0], dtype=float)
        scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
        assert compute_accuracy(scores, labels)  == pytest.approx(1.0)
        assert compute_precision(scores, labels) == pytest.approx(1.0)
        assert compute_recall(scores, labels)    == pytest.approx(1.0)
        assert compute_f1(scores, labels)        == pytest.approx(1.0)
        assert compute_auc(scores, labels)       == pytest.approx(1.0)

    def test_random_classifier(self):
        rng = np.random.default_rng(42)
        labels = (rng.random(1000) > 0.5).astype(float)
        scores = rng.random(1000)
        auc = compute_auc(scores, labels)
        # AUC of random classifier should be ~0.5
        assert 0.4 < auc < 0.6

    def test_all_positive_predictions(self):
        labels = np.array([1, 0, 1, 0], dtype=float)
        scores = np.ones(4) * 0.9  # All above threshold
        tp, fp, fn, tn = 2, 2, 0, 0
        assert compute_precision(scores, labels) == pytest.approx(0.5)
        assert compute_recall(scores, labels)    == pytest.approx(1.0)

    def test_no_positives_edge_case(self):
        labels = np.zeros(10, dtype=float)
        scores = np.random.rand(10)
        # AUC should return 0.5 (undefined, guarded)
        auc = compute_auc(scores, labels)
        assert auc == pytest.approx(0.5)

    def test_f1_zero_division(self):
        labels = np.ones(10, dtype=float)
        scores = np.zeros(10)   # All below threshold → TP=0, FP=0
        f1 = compute_f1(scores, labels)
        assert f1 == pytest.approx(0.0)

    def test_threshold_sensitivity(self):
        labels = np.array([1, 1, 0, 0], dtype=float)
        scores = np.array([0.6, 0.4, 0.3, 0.7])
        # At τ=0.5: preds=[1,0,0,1] → TP=1, FP=1, FN=1
        f1_05 = compute_f1(scores, labels, threshold=0.5)
        # At τ=0.35: preds=[1,1,0,1] → TP=2, FP=1, FN=0
        f1_35 = compute_f1(scores, labels, threshold=0.35)
        assert f1_35 > f1_05

    def test_all_metrics_dict_keys(self):
        labels = np.array([1, 0, 1, 0], dtype=float)
        scores = np.array([0.8, 0.3, 0.7, 0.2])
        result = compute_all_detection_metrics(scores, labels)
        required = {"accuracy", "precision", "recall", "f1", "auc_roc"}
        assert required.issubset(set(result.keys()))


class TestOperationalMetrics:
    def test_wlr_no_loss(self):
        # Perfect intervention: system_loss = 0
        assert compute_wlr(100.0, 0.0) == pytest.approx(1.0)

    def test_wlr_no_improvement(self):
        # No improvement: system_loss == baseline_loss
        assert compute_wlr(100.0, 100.0) == pytest.approx(0.0)

    def test_wlr_zero_baseline(self):
        # Guard against division by zero
        assert compute_wlr(0.0, 0.0) == pytest.approx(0.0)

    def test_wlr_clipped(self):
        # System loss > baseline is clipped to 0
        assert compute_wlr(10.0, 20.0) == pytest.approx(0.0)

    def test_pcr_all_approved(self):
        assert compute_pcr(1000, 1000) == pytest.approx(1.0)

    def test_pcr_none_approved(self):
        assert compute_pcr(0, 100) == pytest.approx(0.0)

    def test_pcr_zero_total(self):
        assert compute_pcr(0, 0) == pytest.approx(1.0)

    def test_rt_basic(self):
        detect = [10, 20, 30]
        respond = [15, 25, 35]
        rt = compute_rt(detect, respond, time_step_seconds=1.0)
        assert rt == pytest.approx(5.0)

    def test_rt_empty(self):
        assert compute_rt([], []) == pytest.approx(0.0)

    def test_rt_scaled(self):
        detect = [0]
        respond = [10]
        # At 5s per step → RT = 50s
        assert compute_rt(detect, respond, time_step_seconds=5.0) == pytest.approx(50.0)
