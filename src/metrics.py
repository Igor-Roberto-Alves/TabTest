"""Métricas de avaliação por tipo de tarefa."""

import numpy as np
from sklearn import metrics

from src.models.base import Prediction


def evaluate(kind: str, y_true: np.ndarray, pred: Prediction) -> dict[str, float]:
    if kind == "regression":
        return {
            "rmse": metrics.root_mean_squared_error(y_true, pred.y_pred),
            "mae": metrics.mean_absolute_error(y_true, pred.y_pred),
            "r2": metrics.r2_score(y_true, pred.y_pred),
        }

    result = {
        "accuracy": metrics.accuracy_score(y_true, pred.y_pred),
        "balanced_acc": metrics.balanced_accuracy_score(y_true, pred.y_pred),
    }
    if pred.proba is not None:
        labels = np.arange(pred.proba.shape[1])
        result["log_loss"] = metrics.log_loss(y_true, pred.proba, labels=labels)
        if kind == "binary":
            result["roc_auc"] = metrics.roc_auc_score(y_true, pred.proba[:, 1])
        else:
            result["roc_auc"] = metrics.roc_auc_score(y_true, pred.proba, multi_class="ovr", labels=labels)
    return result
