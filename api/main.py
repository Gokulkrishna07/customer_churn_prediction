"""
Production FastAPI application for churn prediction.
Ready for deployment to cloud VMs.
"""

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
import sys
import os
import logging
from datetime import datetime
import uvicorn

# Add training directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import modules before loading model
import training.preprocessing
import training.data_loader
from training.final_model import ProductionChurnModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Production API for predicting customer churn",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
model = None

# Request/Response Models
class CustomerData(BaseModel):
    """Customer data schema for prediction request."""
    gender: str = Field(..., description="Gender: Male, Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="0 or 1")
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: str = Field(..., description="Yes or No")
    MultipleLines: str = Field(..., description="Yes, No, or No phone service")
    InternetService: str = Field(..., description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Yes, No, or No internet service")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Yes or No")
    PaymentMethod: str = Field(..., description="Payment method type")
    MonthlyCharges: float = Field(..., gt=0, description="Monthly charges in dollars")
    TotalCharges: float = Field(..., ge=0, description="Total charges in dollars")
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('Gender must be Male or Female')
        return v
    
    @validator('Contract')
    def validate_contract(cls, v):
        if v not in ['Month-to-month', 'One year', 'Two year']:
            raise ValueError('Invalid contract type')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 89.15,
                "TotalCharges": 2139.60
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response schema."""
    customer_id: Optional[str] = None
    churn_prediction: int = Field(..., description="0 = No Churn, 1 = Churn")
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request schema."""
    customers: List[CustomerData] = Field(..., max_items=100)


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    model_version: str
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global model
    try:
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "training",
            "models",
            "production",
            "churn_model_production.pkl"
        )
        model = ProductionChurnModel.load(model_path)
        logger.info(f"✓ Model loaded successfully from {model_path}")
        logger.info(f"Model threshold: {model.threshold}")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


# Helper functions
def get_risk_level(probability: float) -> str:
    """Determine risk level from probability."""
    if probability < 0.3:
        return "LOW"
    elif probability < 0.6:
        return "MEDIUM"
    else:
        return "HIGH"


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Telco Churn Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint for load balancers."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version=model.metadata.get('model_type', 'unknown') if model else 'none',
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_churn(
    customer: CustomerData,
    api_key: Optional[str] = Header(None, description="API Key for authentication")
):
    """
    Predict churn for a single customer.
    
    Returns:
        Prediction with probability and risk level
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # In production, validate API key here
    # if not validate_api_key(api_key):
    #     raise HTTPException(status_code=401, detail="Invalid API key")
    
    try:
        # Make prediction
        result = model.predict_single(customer.dict())
        
        # Log prediction (in production, log to database)
        logger.info(f"Prediction made: churn_prob={result['churn_probability']:.3f}")
        
        return PredictionResponse(
            churn_prediction=result['churn_prediction'],
            churn_probability=result['churn_probability'],
            risk_level=get_risk_level(result['churn_probability']),
            confidence=result['confidence'],
            timestamp=datetime.utcnow().isoformat(),
            model_version=model.metadata.get('model_type', 'unknown')
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(
    request: BatchPredictionRequest,
    api_key: Optional[str] = Header(None, description="API Key for authentication")
):
    """
    Predict churn for multiple customers (max 100).
    
    Returns:
        List of predictions
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.customers) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 customers per batch")
    
    try:
        results = []
        for customer in request.customers:
            result = model.predict_single(customer.dict())
            results.append(PredictionResponse(
                churn_prediction=result['churn_prediction'],
                churn_probability=result['churn_probability'],
                risk_level=get_risk_level(result['churn_probability']),
                confidence=result['confidence'],
                timestamp=datetime.utcnow().isoformat(),
                model_version=model.metadata.get('model_type', 'unknown')
            ))
        
        logger.info(f"Batch prediction completed: {len(results)} customers")
        return {"predictions": results, "count": len(results)}
    
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info", tags=["Model"])
async def model_info():
    """Get information about the current model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model.metadata.get('model_type', 'unknown'),
        "threshold": model.threshold,
        "features": len(model.preprocessor.feature_names),
        "top_features": list(model.get_feature_importance(5).keys()),
        "created_at": model.metadata.get('created_at', 'unknown')
    }


# Run the application
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        workers=4  # Adjust based on CPU cores
    )