import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    model_name: str
    precision: float
    recall: float
    f1: float
    roc_auc: float
    avg_precision: float
    specificity: float
    confusion_matrix: list[list[int]] = field(default_factory=list)

    def to_loggable_dict(self) -> dict[str, float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "avg_precision": self.avg_precision,
            "specificity": self.specificity,
        }


class ModelEvaluator:
    def __init__(self, model: object, model_name: str = "Model") -> None:
        self.model = model
        self.model_name = model_name
        self.metrics: EvaluationMetrics | None = None

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationMetrics:
        y_pred: np.ndarray = self.model.predict(X_test)  # type: ignore[attr-defined]
        y_prob: np.ndarray = self.model.predict_proba(X_test)[:, 1]  # type: ignore[attr-defined]
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        self.metrics = EvaluationMetrics(
            model_name=self.model_name,
            precision=float(precision_score(y_test, y_pred)),
            recall=float(recall_score(y_test, y_pred)),
            f1=float(f1_score(y_test, y_pred)),
            roc_auc=float(roc_auc_score(y_test, y_prob)),
            avg_precision=float(average_precision_score(y_test, y_prob)),
            specificity=float(tn / (tn + fp)),
            confusion_matrix=cm.tolist(),
        )
        logger.info(
            "%s — F1=%.4f, ROC-AUC=%.4f, Recall=%.4f",
            self.model_name, self.metrics.f1, self.metrics.roc_auc, self.metrics.recall,
        )
        return self.metrics

    def find_optimal_threshold(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> tuple[float, float]:
        y_prob: np.ndarray = self.model.predict_proba(X_test)[:, 1]  # type: ignore[attr-defined]
        thresholds = np.arange(0.1, 0.9, 0.02)
        f1_scores = [
            f1_score(y_test, (y_prob >= t).astype(int)) for t in thresholds
        ]
        optimal_idx = int(np.argmax(f1_scores))
        optimal_threshold = float(thresholds[optimal_idx])
        optimal_f1 = float(f1_scores[optimal_idx])
        logger.info("Optimal threshold: %.2f (F1=%.4f)", optimal_threshold, optimal_f1)
        return optimal_threshold, optimal_f1

    def save_metrics(self, output_dir: str = "results/evaluation") -> None:
        if self.metrics is None:
            raise ValueError("No metrics to save. Call evaluate() first.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(output_dir) / f"{self.model_name.replace(' ', '_')}_{timestamp}"
        out_path.mkdir(parents=True, exist_ok=True)
        metrics_path = out_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(
                {**self.metrics.to_loggable_dict(), "confusion_matrix": self.metrics.confusion_matrix},
                f, indent=4,
            )
        logger.info("Metrics saved: path=%s", metrics_path)
