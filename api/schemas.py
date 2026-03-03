from pydantic import BaseModel, Field, field_validator


class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v: str) -> str:
        if v not in ("Male", "Female"):
            raise ValueError("gender must be Male or Female")
        return v

    @field_validator("Contract")
    @classmethod
    def validate_contract(cls, v: str) -> str:
        valid = ("Month-to-month", "One year", "Two year")
        if v not in valid:
            raise ValueError(f"Contract must be one of {valid}")
        return v

    model_config = {
        "json_schema_extra": {
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
                "TotalCharges": 2139.60,
            }
        }
    }


class PredictionResponse(BaseModel):
    churn_prediction: int
    churn_probability: float
    risk_level: str
    confidence: float
    model_type: str


class BatchPredictionRequest(BaseModel):
    customers: list[CustomerData] = Field(..., max_length=100)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
