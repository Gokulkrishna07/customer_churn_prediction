import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.preprocessor import TelcoPreprocessor

logger = logging.getLogger(__name__)


class ProductionChurnModel:
    def __init__(self, model: object, preprocessor: TelcoPreprocessor, threshold: float = 0.5) -> None:
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.metadata: dict[str, object] = {
            "created_at": datetime.now().isoformat(),
            "threshold": threshold,
            "model_type": type(model).__name__,
            "feature_names": preprocessor.feature_names,
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)  # type: ignore[attr-defined]

    def predict_single(self, customer_data: dict[str, object]) -> dict[str, object]:
        df = pd.DataFrame([customer_data])
        df = self.preprocessor.clean_data(df)
        df = self.preprocessor.engineer_features(df)
        df = self.preprocessor.encode_features(df, fit=False)
        df = self.preprocessor.scale_features(df, fit=False)

        if "Churn" in df.columns:
            df = df.drop("Churn", axis=1)

        for feature in set(self.preprocessor.feature_names) - set(df.columns):
            df[feature] = 0
        df = df[self.preprocessor.feature_names]

        proba: np.ndarray = self.predict_proba(df)[0]
        prediction = int(self.predict(df)[0])

        return {
            "churn_prediction": prediction,
            "churn_probability": float(proba[1]),
            "no_churn_probability": float(proba[0]),
            "confidence": float(abs(proba[1] - self.threshold) / (1 - self.threshold)),
        }

    def get_feature_importance(self, top_n: int = 10) -> dict[str, float]:
        if not hasattr(self.model, "feature_importances_"):
            return {}
        importances: np.ndarray = self.model.feature_importances_  # type: ignore[attr-defined]
        indices = np.argsort(importances)[::-1][:top_n]
        return {self.preprocessor.feature_names[i]: float(importances[i]) for i in indices}

    def save(self, path: str = "models/production") -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self, Path(path) / "churn_model_production.pkl")
        with open(Path(path) / "model_metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=4)
        with open(Path(path) / "feature_importance.json", "w") as f:
            json.dump(self.get_feature_importance(20), f, indent=4)
        logger.info("Production model saved: path=%s", path)

    @classmethod
    def load(cls, path: str = "models/production/churn_model_production.pkl") -> "ProductionChurnModel":
        model: ProductionChurnModel = joblib.load(path)
        logger.info("Production model loaded: path=%s", path)
        return model


def create_production_model(
    model_path: str,
    preprocessor_path: str,
    optimal_threshold: float = 0.38,
) -> ProductionChurnModel:
    model = joblib.load(model_path)
    preprocessor: TelcoPreprocessor = joblib.load(preprocessor_path)
    prod_model = ProductionChurnModel(model, preprocessor, optimal_threshold)
    logger.info(
        "Production model created: type=%s, threshold=%.2f, features=%d",
        type(model).__name__, optimal_threshold, len(preprocessor.feature_names),
    )
    return prod_model
