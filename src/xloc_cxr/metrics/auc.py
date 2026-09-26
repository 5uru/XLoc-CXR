"""Per-class ROC AUC for multi-label classification (pure numpy/scipy)."""
import numpy as np
from scipy.stats import rankdata


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC via the Mann-Whitney U statistic (handles ties).

    Args:
        scores: (N,) predicted probabilities for one class
        labels: (N,) binary ground truth for that class

    Returns:
        AUC in [0, 1], or NaN if only one class present
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    ranks = rankdata(np.concatenate([pos, neg]))
    auc = (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return float(auc)


def per_class_auc(probs: np.ndarray, labels: np.ndarray, class_names: list) -> dict:
    """Compute AUC for every class + macro average.

    Args:
        probs: (N, C) sigmoid probabilities
        labels: (N, C) binary ground truth
        class_names: list of C class names

    Returns:
        dict class_name → AUC, plus "MACRO" key
    """
    out = {}
    for i, name in enumerate(class_names):
        out[name] = roc_auc(probs[:, i], labels[:, i])
    valid = [v for v in out.values() if not np.isnan(v)]
    out["MACRO"] = float(np.mean(valid)) if valid else float("nan")
    return out
