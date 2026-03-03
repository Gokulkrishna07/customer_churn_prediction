import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException

from api.predictor import get_model, get_risk_level
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerData,
    HealthResponse,
    PredictionResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    get_model()
    logger.info("Model loaded and ready")
    yield


app = FastAPI(
    title="Telco Churn Prediction API",
    version="1.0.0",
    description="Production churn prediction service",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        model = get_model()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_type=str(model.metadata.get("model_type", "unknown")),
        )
    except Exception:
        return HealthResponse(status="degraded", model_loaded=False, model_type="none")


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData) -> PredictionResponse:
    try:
        model = get_model()
    except Exception as exc:
        logger.exception("Model unavailable")
        raise HTTPException(status_code=503, detail="Model not available") from exc

    try:
        result = model.predict_single(customer.model_dump())
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    return PredictionResponse(
        churn_prediction=int(result["churn_prediction"]),  # type: ignore[arg-type]
        churn_probability=float(result["churn_probability"]),  # type: ignore[arg-type]
        risk_level=get_risk_level(float(result["churn_probability"])),  # type: ignore[arg-type]
        confidence=float(result["confidence"]),  # type: ignore[arg-type]
        model_type=str(model.metadata.get("model_type", "unknown")),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    try:
        model = get_model()
    except Exception as exc:
        logger.exception("Model unavailable")
        raise HTTPException(status_code=503, detail="Model not available") from exc

    predictions: list[PredictionResponse] = []
    try:
        for customer in request.customers:
            result = model.predict_single(customer.model_dump())
            predictions.append(
                PredictionResponse(
                    churn_prediction=int(result["churn_prediction"]),  # type: ignore[arg-type]
                    churn_probability=float(result["churn_probability"]),  # type: ignore[arg-type]
                    risk_level=get_risk_level(float(result["churn_probability"])),  # type: ignore[arg-type]
                    confidence=float(result["confidence"]),  # type: ignore[arg-type]
                    model_type=str(model.metadata.get("model_type", "unknown")),
                )
            )
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Batch prediction failed") from exc

    logger.info("Batch prediction: count=%d", len(predictions))
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
