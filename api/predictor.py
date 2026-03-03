import logging
import os
from functools import lru_cache

from src.models.production import ProductionChurnModel

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_model() -> ProductionChurnModel:
    model_path = os.environ.get(
        "MODEL_PATH", "models/production/churn_model_production.pkl"
    )
    model = ProductionChurnModel.load(model_path)
    logger.info("Production model loaded: path=%s, type=%s", model_path, model.metadata.get("model_type"))
    return model


def get_risk_level(probability: float) -> str:
    if probability < 0.3:
        return "LOW"
    if probability < 0.6:
        return "MEDIUM"
    return "HIGH"
