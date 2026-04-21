"""
src/evaluation/metrics.py
--------------------------
Evaluation metrics for AquaAgent.

Paper Table 1 (primary metrics):
  Accuracy, Precision, Recall, F1-Score, AUC-ROC  — ADA detection quality
  Water Loss Reduction (WLR%)                       — Hydraulic effectiveness
  Policy Compliance Rate (PCR%)                     — Governance adherence
  Mean Corrective Response Time (RT, seconds)       — Operational speed
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


# ------------------------------------------------------------------
# ADA detection metrics
# ------------------------------------------------------------------

def compute_confusion(scores: np.ndarray, labels: np.ndarray,
                      threshold: float = 0.5) -> Tuple[int, int, int, int]:
    """
    Compute binary confusion matrix.

    Args:
        scores:    Predicted anomaly probabilities [N].
        labels:    Ground-truth binary labels [N].
        threshold: Decision threshold τ.

    Returns:
        (TP, FP, FN, TN)
    """
    preds = (scores >= threshold).astype(int)
    gt    = labels.astype(int)
    tp = int(np.sum((preds == 1) & (gt == 1)))
    fp = int(np.sum((preds == 1) & (gt == 0)))
    fn = int(np.sum((preds == 0) & (gt == 1)))
    tn = int(np.sum((preds == 0) & (gt == 0)))
    return tp, fp, fn, tn


def compute_accuracy(scores: np.ndarray, labels: np.ndarray,
                     threshold: float = 0.5) -> float:
    """Overall accuracy."""
    tp, fp, fn, tn = compute_confusion(scores, labels, threshold)
    total = tp + fp + fn + tn
    return (tp + tn) / total if total > 0 else 0.0


def compute_precision(scores: np.ndarray, labels: np.ndarray,
                      threshold: float = 0.5) -> float:
    """Precision = TP / (TP + FP)."""
    tp, fp, _, _ = compute_confusion(scores, labels, threshold)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def compute_recall(scores: np.ndarray, labels: np.ndarray,
                   threshold: float = 0.5) -> float:
    """Recall = TP / (TP + FN)."""
    tp, _, fn, _ = compute_confusion(scores, labels, threshold)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def compute_f1(scores: np.ndarray, labels: np.ndarray,
               threshold: float = 0.5) -> float:
    """F1 = 2 · (Precision · Recall) / (Precision + Recall)."""
    p = compute_precision(scores, labels, threshold)
    r = compute_recall(scores, labels, threshold)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def compute_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    AUC-ROC via trapezoidal integration over the ROC curve.
    Handles edge cases (no positives, no negatives).
    """
    labels = labels.astype(int)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by descending score
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Accumulate TPR and FPR
    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Prepend origin
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])

    # np.trapz was renamed to np.trapezoid in NumPy 2.0; support both
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    return float(_trapz(tpr, fpr))


def compute_all_detection_metrics(scores: np.ndarray,
                                   labels: np.ndarray,
                                   threshold: float = 0.5) -> Dict[str, float]:
    """Compute all ADA detection metrics in one call."""
    return {
        "accuracy":  compute_accuracy(scores, labels, threshold),
        "precision": compute_precision(scores, labels, threshold),
        "recall":    compute_recall(scores, labels, threshold),
        "f1":        compute_f1(scores, labels, threshold),
        "auc_roc":   compute_auc(scores, labels),
    }


# ------------------------------------------------------------------
# Water Loss Reduction (WLR)
# ------------------------------------------------------------------

def compute_wlr(baseline_loss_L: float, system_loss_L: float) -> float:
    """
    Water Loss Reduction (WLR%) — paper primary metric.

    WLR = (baseline_loss − system_loss) / baseline_loss × 100%

    Args:
        baseline_loss_L: Water loss (L) without any intervention.
        system_loss_L:   Water loss (L) with AquaAgent active.

    Returns:
        WLR as a fraction in [0, 1].  Multiply by 100 for percentage.
    """
    if baseline_loss_L <= 0:
        return 0.0
    reduction = (baseline_loss_L - system_loss_L) / baseline_loss_L
    return float(np.clip(reduction, 0.0, 1.0))


def compute_wlr_rolling(baseline_losses: np.ndarray,
                         system_losses: np.ndarray,
                         window: int = 30 * 86400) -> np.ndarray:
    """
    Compute rolling WLR over a time series (for Fig. 3 reproduction).

    Args:
        baseline_losses: Cumulative loss per step under no-action baseline [T].
        system_losses:   Cumulative loss per step under AquaAgent [T].
        window:          Rolling window size in steps (default: 30 days at 1Hz).

    Returns:
        rolling_wlr: WLR at each time step [T].
    """
    T = len(baseline_losses)
    rolling = np.zeros(T, dtype=np.float32)
    for t in range(T):
        t0 = max(0, t - window + 1)
        b = float(baseline_losses[t0:t+1].sum())
        s = float(system_losses[t0:t+1].sum())
        rolling[t] = compute_wlr(b, s)
    return rolling


# ------------------------------------------------------------------
# Policy Compliance Rate (PCR)
# ------------------------------------------------------------------

def compute_pcr(n_approved: int, n_total: int) -> float:
    """
    Policy Compliance Rate (PCR) = fraction of steps where governance
    approved the DA action without override.

    Paper Table 1: AquaAgent PCR = 90.9%

    Args:
        n_approved: Number of steps with no governance override.
        n_total:    Total number of decision steps.

    Returns:
        PCR as a fraction in [0, 1].
    """
    return n_approved / n_total if n_total > 0 else 1.0


# ------------------------------------------------------------------
# Mean Corrective Response Time (RT)
# ------------------------------------------------------------------

def compute_rt(detection_steps: List[int],
               response_steps: List[int],
               time_step_seconds: float = 1.0) -> float:
    """
    Mean Corrective Response Time in seconds.

    RT = mean(response_step − detection_step) × time_step_seconds

    Paper Table 1: AquaAgent RT = 48.3 seconds

    Args:
        detection_steps: Time step when each leak was first flagged.
        response_steps:  Time step when a corrective action was taken.
        time_step_seconds: Duration of each time step (default: 1s).

    Returns:
        Mean RT in seconds.
    """
    if not detection_steps or not response_steps:
        return 0.0
    n = min(len(detection_steps), len(response_steps))
    delays = [
        max(0, (response_steps[i] - detection_steps[i]) * time_step_seconds)
        for i in range(n)
    ]
    return float(np.mean(delays)) if delays else 0.0


# ------------------------------------------------------------------
# Summary metric bundle
# ------------------------------------------------------------------

def compute_summary_metrics(scores: np.ndarray,
                              labels: np.ndarray,
                              threshold: float = 0.5,
                              baseline_loss: float = 1.0,
                              system_loss: float = 0.0,
                              n_approved: int = 1,
                              n_total: int = 1,
                              detection_steps: Optional[List[int]] = None,
                              response_steps: Optional[List[int]] = None) -> Dict[str, float]:
    """Compute all paper metrics in one call."""
    det = compute_all_detection_metrics(scores, labels, threshold)
    det["wlr"]  = compute_wlr(baseline_loss, system_loss)
    det["pcr"]  = compute_pcr(n_approved, n_total)
    det["rt_s"] = compute_rt(
        detection_steps or [], response_steps or []
    )
    return det
