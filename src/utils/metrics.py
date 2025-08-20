# src/utils/metrics.py
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score


def compute_metrics(y_true, y_probs, threshold: float = 0.5):
    """
    y_true (list/np.array): истинные метки (0/1)
    y_probs (list/np.array): вероятности от модели (sigmoid(logits))
    threshold (float): порог для binarization (по умолчанию 0.5)
    """
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    # бинаризация по порогу
    y_pred = (y_probs >= threshold).astype(int)

    metrics = {}

    try:
        metrics['roc_auc'] = roc_auc_score(y_true, y_probs)
    except ValueError:
        metrics['roc_auc'] = float('nan')  # если в y_true только один класс

    try:
        metrics['ap'] = average_precision_score(y_true, y_probs)
    except ValueError:
        metrics['ap'] = float('nan')

    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)

    return metrics
