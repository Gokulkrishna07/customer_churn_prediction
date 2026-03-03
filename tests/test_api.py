import os
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

SAMPLE_CUSTOMER = {
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


@pytest.fixture()
def mock_model() -> MagicMock:
    model = MagicMock()
    model.metadata = {"model_type": "RandomForestClassifier"}
    model.predict_single.return_value = {
        "churn_prediction": 1,
        "churn_probability": 0.75,
        "no_churn_probability": 0.25,
        "confidence": 0.5,
    }
    return model


@pytest.fixture()
def client(mock_model: MagicMock) -> TestClient:
    with patch("api.predictor.get_model", return_value=mock_model):
        from api.main import app
        return TestClient(app)


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_schema(client: TestClient) -> None:
    data = client.get("/health").json()
    assert "status" in data
    assert "model_loaded" in data
    assert "model_type" in data


def test_predict_returns_200(client: TestClient) -> None:
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200


def test_predict_response_schema(client: TestClient) -> None:
    data = client.post("/predict", json=SAMPLE_CUSTOMER).json()
    assert "churn_prediction" in data
    assert "churn_probability" in data
    assert "risk_level" in data
    assert "confidence" in data


def test_predict_probability_range(client: TestClient) -> None:
    data = client.post("/predict", json=SAMPLE_CUSTOMER).json()
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_risk_level_values(client: TestClient) -> None:
    data = client.post("/predict", json=SAMPLE_CUSTOMER).json()
    assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_invalid_gender_returns_422(client: TestClient) -> None:
    payload = {**SAMPLE_CUSTOMER, "gender": "Unknown"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_contract_returns_422(client: TestClient) -> None:
    payload = {**SAMPLE_CUSTOMER, "Contract": "Weekly"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_batch_predict_returns_200(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"customers": [SAMPLE_CUSTOMER]})
    assert response.status_code == 200


def test_batch_predict_count(client: TestClient) -> None:
    payload = {"customers": [SAMPLE_CUSTOMER, SAMPLE_CUSTOMER]}
    data = client.post("/predict/batch", json=payload).json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
