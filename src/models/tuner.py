import logging
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_PARAM_GRIDS: dict[str, dict[str, dict[str, object]]] = {
    "Random Forest": {
        "grid": {
            "n_estimators": [100, 200, 300],
            "max_depth": [10, 15, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "class_weight": ["balanced", "balanced_subsample"],
        },
        "random": {
            "n_estimators": randint(100, 500),
            "max_depth": [10, 15, 20, 25, 30, None],
            "min_samples_split": randint(2, 20),
            "min_samples_leaf": randint(1, 10),
            "max_features": ["sqrt", "log2", None],
            "class_weight": ["balanced", "balanced_subsample"],
        },
    },
    "XGBoost": {
        "grid": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [3, 5, 7],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        },
        "random": {
            "n_estimators": randint(100, 500),
            "learning_rate": uniform(0.01, 0.2),
            "max_depth": randint(3, 10),
            "subsample": uniform(0.7, 0.3),
            "colsample_bytree": uniform(0.7, 0.3),
        },
    },
    "LightGBM": {
        "grid": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [5, 10, 15],
            "num_leaves": [31, 50, 70],
        },
        "random": {
            "n_estimators": randint(100, 500),
            "learning_rate": uniform(0.01, 0.2),
            "max_depth": randint(5, 20),
            "num_leaves": randint(20, 100),
        },
    },
}


class HyperparameterTuner:
    def __init__(self, random_state: int = 42, n_jobs: int = -1) -> None:
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params: dict[str, dict[str, object]] = {}
        self.best_models: dict[str, object] = {}

    def _get_base_model(self, model_name: str) -> object:
        if model_name == "Random Forest":
            return RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        if model_name == "Gradient Boosting":
            return GradientBoostingClassifier(random_state=self.random_state)
        if model_name == "XGBoost":
            return XGBClassifier(
                random_state=self.random_state, eval_metric="logloss", n_jobs=self.n_jobs
            )
        if model_name == "LightGBM":
            return LGBMClassifier(
                random_state=self.random_state, verbose=-1, n_jobs=self.n_jobs
            )
        raise ValueError(f"Unknown model: {model_name}")

    def tune_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        search_type: str = "random",
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = "f1",
    ) -> tuple[object, dict[str, object]]:
        base_model = self._get_base_model(model_name)
        param_grid = _PARAM_GRIDS[model_name][search_type]
        cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scorer = make_scorer(f1_score) if scoring == "f1" else make_scorer(roc_auc_score, needs_proba=True)

        if search_type == "grid":
            search: GridSearchCV | RandomizedSearchCV = GridSearchCV(
                base_model, param_grid, cv=cv_strategy,  # type: ignore[arg-type]
                scoring=scorer, n_jobs=self.n_jobs, verbose=1,
            )
        else:
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter,  # type: ignore[arg-type]
                cv=cv_strategy, scoring=scorer, n_jobs=self.n_jobs,
                verbose=1, random_state=self.random_state,
            )

        search.fit(X_train, y_train)
        self.best_params[model_name] = search.best_params_
        self.best_models[model_name] = search.best_estimator_
        logger.info("%s best CV score: %.4f", model_name, search.best_score_)
        return search.best_estimator_, search.best_params_

    def save_results(self, path: str = "models/tuned") -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        for model_name, model in self.best_models.items():
            safe_name = model_name.replace(" ", "_").lower()
            joblib.dump(model, Path(path) / f"{safe_name}_tuned.pkl")
            logger.info("Saved tuned model: %s", safe_name)
