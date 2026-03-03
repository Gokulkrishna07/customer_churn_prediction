import logging
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import TelcoDataLoader
from src.evaluation.evaluator import ModelEvaluator
from src.features.preprocessor import TelcoPreprocessor
from src.models.production import create_production_model
from src.models.trainer import ChurnModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run(config_path: str = "configs/config.yaml") -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    experiment_name = cfg["mlflow"]["experiment_name"]

    logger.info("Step 1/5: Loading data")
    loader = TelcoDataLoader(data_path=cfg["data"]["raw_path"])
    df = loader.load_data()

    logger.info("Step 2/5: Preprocessing")
    preprocessor = TelcoPreprocessor()
    X, y, _ = preprocessor.prepare_features(df, fit=True)

    preprocessor_path = Path(cfg["model"]["preprocessor_path"])
    preprocessor.save(str(preprocessor_path))

    logger.info("Step 3/5: Training all models")
    trainer = ChurnModelTrainer(
        random_state=cfg["training"]["random_seed"],
        mlflow_tracking_uri=mlflow_uri,
        experiment_name=experiment_name,
    )
    X_train, X_test, y_train, y_test = trainer.split_data(
        X, y, test_size=cfg["training"]["test_size"]
    )
    trainer.train_all_models(
        X_train, X_test, y_train, y_test,
        imbalance_method=cfg["training"]["imbalance_method"],
        cv=cfg["training"]["cross_validate"],
        log_to_mlflow=True,
    )

    logger.info("Step 4/5: Evaluating best model")
    evaluator = ModelEvaluator(trainer.best_model, trainer.best_model_name or "best_model")
    evaluator.evaluate(X_test, y_test)
    optimal_threshold, _ = evaluator.find_optimal_threshold(X_test, y_test)
    evaluator.save_metrics(cfg["model"]["results_path"])

    logger.info("Step 5/5: Saving production model")
    best_safe_name = trainer.best_model_name.replace(" ", "_").lower()  # type: ignore[union-attr]
    model_path = Path(cfg["model"]["save_path"]) / f"{best_safe_name}.pkl"
    trainer.save_model(path=cfg["model"]["save_path"])

    prod_model = create_production_model(
        model_path=str(model_path),
        preprocessor_path=str(preprocessor_path),
        optimal_threshold=optimal_threshold,
    )
    prod_model.save(path=cfg["model"]["production_path"])

    logger.info(
        "Pipeline complete — best_model=%s, threshold=%.2f",
        trainer.best_model_name, optimal_threshold,
    )


if __name__ == "__main__":
    run()
