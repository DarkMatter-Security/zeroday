"""
builds/shared/utils.py — Shared utilities for H-ANS builds.

Common functions for visualization, metrics, and analysis across builds.
"""

import math
import torch


def entropy(probs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute Shannon entropy H(p) = -Σ p_i log(p_i).

    Args:
        probs: Tensor of shape (..., n_categories) — probability distribution.
        eps: Small constant for numerical stability.

    Returns:
        Entropy tensor of shape (...).
    """
    p = probs + eps
    return -torch.sum(p * torch.log(p), dim=-1)


def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute KL divergence KL(p || q).

    Args:
        p: First distribution (..., n_categories).
        q: Second distribution (..., n_categories).
        eps: Stability constant.

    Returns:
        KL divergence tensor of shape (...).
    """
    p = p + eps
    q = q + eps
    return torch.sum(p * torch.log(p / q), dim=-1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence JSD(p || q).

    Args:
        p, q: Distributions.

    Returns:
        JSD value.
    """
    m = (p + q) / 2
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def resonance_curve(fold_gaps: torch.Tensor,
                    s_range: tuple[float, float] = (0.1, 10.0),
                    n_points: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the fractal resonance curve R(S) for a range of S values.

    Args:
        fold_gaps: Tensor of shape (4,) — the four Δ_f.
        s_range: (min, max) for S values.
        n_points: Number of points.

    Returns:
        Tuple (S_values, R_values) for plotting.
    """
    s = torch.linspace(s_range[0], s_range[1], n_points)
    r = torch.ones_like(s)
    for f in fold_gaps:
        r = r * (s ** 2 / (s ** 2 + f ** 2))
    return s, r


def confusion_matrix(preds: torch.Tensor,
                     targets: torch.Tensor,
                     n_classes: int) -> torch.Tensor:
    """
    Compute confusion matrix.

    Args:
        preds: Predicted class indices (N,).
        targets: Target class indices (N,).
        n_classes: Number of classes.

    Returns:
        Confusion matrix (n_classes, n_classes).
    """
    cm = torch.zeros(n_classes, n_classes, dtype=torch.long)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm


def detection_metrics(scores: torch.Tensor,
                      labels: torch.Tensor,
                      threshold: float = 0.5) -> dict:
    """
    Compute binary detection metrics.

    Args:
        scores: Anomaly scores (N,).
        labels: Ground truth (0 = normal, 1 = anomaly) (N,).
        threshold: Decision threshold.

    Returns:
        Dict with accuracy, precision, recall, F1, FPR, TPR.
    """
    preds = (scores > threshold).long()
    labels = labels.long()

    tp = ((preds == 1) & (labels == 1)).sum().float()
    tn = ((preds == 0) & (labels == 0)).sum().float()
    fp = ((preds == 1) & (labels == 0)).sum().float()
    fn = ((preds == 0) & (labels == 1)).sum().float()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    tpr = recall

    return {
        'accuracy': accuracy.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'fpr': fpr.item(),
        'tpr': tpr.item(),
    }
