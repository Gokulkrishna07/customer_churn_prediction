import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    def __init__(
        self,
        random_state: int = 42,
        mlflow_tracking_uri: str | None = None,
        experiment_name: str = "churn-prediction",
    ) -> None:
        self.random_state = random_state
        self.models: dict[str, object] = {}
        self.results: dict[str, dict[str, object]] = {}
        self.best_model: object = None
        self.best_model_name: str | None = None

        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow tracking URI: %s", mlflow_tracking_uri)

    def get_models(self) -> dict[str, object]:
        return {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, random_state=self.random_state, class_weight="balanced"
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=10,
                min_samples_leaf=4, random_state=self.random_state,
                class_weight="balanced", n_jobs=-1,
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=self.random_state,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=self.random_state, eval_metric="logloss",
            ),
            "LightGBM": LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5,
                random_state=self.random_state, class_weight="balanced", verbose=-1,
            ),
        }

    def split_data(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        logger.info(
            "Split — train: %d, test: %d, train_churn_rate: %.2f%%",
            len(X_train), len(X_test), float(y_train.mean()) * 100,
        )
        return X_train, X_test, y_train, y_test

    def handle_imbalance(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        method: str = "smote",
    ) -> tuple[pd.DataFrame, pd.Series]:
        if method == "none":
            return X_train, y_train
        logger.info("Applying %s resampling: before=%s", method, np.bincount(y_train))
        if method == "smote":
            sampler: SMOTE | RandomUnderSampler = SMOTE(random_state=self.random_state)
        elif method == "undersample":
            sampler = RandomUnderSampler(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown imbalance method: {method}")
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        logger.info("After resampling: %s", np.bincount(y_resampled))
        return X_resampled, y_resampled

    def evaluate_model(
        self, model: object, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> dict[str, object]:
        y_pred = model.predict(X_test)  # type: ignore[attr-defined]
        y_prob = model.predict_proba(X_test)[:, 1]  # type: ignore[attr-defined]
        metrics: dict[str, object] = {
            "model_name": model_name,
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
        }
        logger.info(
            "%s — F1=%.4f, ROC-AUC=%.4f", model_name, metrics["f1_score"], metrics["roc_auc"]
        )
        return metrics

    def cross_validate_model(
        self, model: object, X: pd.DataFrame, y: pd.Series, cv: int = 5
    ) -> dict[str, float]:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores: dict[str, np.ndarray] = {
            metric: cross_val_score(model, X, y, cv=skf, scoring=metric)  # type: ignore[arg-type]
            for metric in ("f1", "roc_auc")
        }
        return {
            f"{m}_mean": float(v.mean()) for m, v in scores.items()
        } | {
            f"{m}_std": float(v.std()) for m, v in scores.items()
        }

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        imbalance_method: str = "smote",
        cv: bool = True,
        log_to_mlflow: bool = True,
    ) -> None:
        X_train_rs, y_train_rs = self.handle_imbalance(X_train, y_train, imbalance_method)
        models = self.get_models()

        for model_name, model in models.items():
            logger.info("Training: %s", model_name)
            if log_to_mlflow:
                with mlflow.start_run(run_name=model_name):
                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("imbalance_method", imbalance_method)
                    mlflow.log_param("random_state", self.random_state)
                    model.fit(X_train_rs, y_train_rs)  # type: ignore[union-attr]
                    self.models[model_name] = model
                    metrics = self.evaluate_model(model, X_test, y_test, model_name)
                    mlflow.log_metrics({
                        k: v for k, v in metrics.items() if isinstance(v, float)
                    })
                    if cv:
                        cv_results = self.cross_validate_model(model, X_train, y_train)
                        metrics["cv_results"] = cv_results
                        mlflow.log_metrics(cv_results)
                    try:
                        if "XGBoost" in model_name:
                            mlflow.xgboost.log_model(model, "model")
                        elif "LightGBM" in model_name:
                            mlflow.lightgbm.log_model(model, "model")
                        else:
                            mlflow.sklearn.log_model(model, "model")
                    except Exception as exc:
                        logger.warning("Could not log model artifact: %s", exc)
                    self.results[model_name] = metrics
            else:
                model.fit(X_train_rs, y_train_rs)  # type: ignore[union-attr]
                self.models[model_name] = model
                metrics = self.evaluate_model(model, X_test, y_test, model_name)
                if cv:
                    metrics["cv_results"] = self.cross_validate_model(model, X_train, y_train)
                self.results[model_name] = metrics

        self._select_best_model()

    def _select_best_model(self) -> None:
        best_f1 = 0.0
        for model_name, metrics in self.results.items():
            f1 = float(metrics["f1_score"])  # type: ignore[arg-type]
            if f1 > best_f1:
                best_f1 = f1
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        logger.info("Best model: %s (F1=%.4f)", self.best_model_name, best_f1)

    def get_results_dataframe(self) -> pd.DataFrame:
        rows = []
        for model_name, metrics in self.results.items():
            row = {
                "Model": model_name,
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"],
            }
            if "cv_results" in metrics:
                cv = metrics["cv_results"]
                row["CV_F1_Mean"] = cv["f1_mean"]  # type: ignore[index]
                row["CV_ROC_AUC_Mean"] = cv["roc_auc_mean"]  # type: ignore[index]
            rows.append(row)
        return pd.DataFrame(rows).sort_values("F1-Score", ascending=False)

    def save_model(self, model_name: str | None = None, path: str = "models") -> None:
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        Path(path).mkdir(parents=True, exist_ok=True)
        safe_name = model_name.replace(" ", "_").lower()  # type: ignore[union-attr]
        joblib.dump(model, Path(path) / f"{safe_name}.pkl")
        logger.info("Saved model: %s", safe_name)
