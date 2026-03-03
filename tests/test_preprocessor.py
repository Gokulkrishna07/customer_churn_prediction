import numpy as np
import pandas as pd
import pytest

from src.features.preprocessor import TelcoPreprocessor


@pytest.fixture()
def raw_df() -> pd.DataFrame:
    return pd.DataFrame({
        "customerID": ["1", "2", "3"],
        "gender": ["Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [2, 24, 60],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "MonthlyCharges": [85.5, 55.0, 20.0],
        "TotalCharges": ["171.0", "1320.0", "  "],
        "Churn": ["Yes", "No", "No"],
    })


@pytest.fixture()
def preprocessor() -> TelcoPreprocessor:
    return TelcoPreprocessor()


def test_clean_data_drops_customer_id(preprocessor: TelcoPreprocessor, raw_df: pd.DataFrame) -> None:
    cleaned = preprocessor.clean_data(raw_df)
    assert "customerID" not in cleaned.columns


def test_clean_data_fixes_total_charges(preprocessor: TelcoPreprocessor, raw_df: pd.DataFrame) -> None:
    cleaned = preprocessor.clean_data(raw_df)
    assert cleaned["TotalCharges"].isnull().sum() == 0
    assert cleaned["TotalCharges"].dtype in (np.float64, np.float32)


def test_clean_data_converts_senior_citizen(
    preprocessor: TelcoPreprocessor, raw_df: pd.DataFrame
) -> None:
    cleaned = preprocessor.clean_data(raw_df)
    assert set(cleaned["SeniorCitizen"].unique()).issubset({"Yes", "No"})


def test_engineer_features_creates_new_columns(
    preprocessor: TelcoPreprocessor, raw_df: pd.DataFrame
) -> None:
    cleaned = preprocessor.clean_data(raw_df)
    engineered = preprocessor.engineer_features(cleaned)
    for col in ("total_services", "has_internet", "has_phone", "charges_ratio", "auto_payment"):
        assert col in engineered.columns


def test_prepare_features_returns_correct_shapes(
    preprocessor: TelcoPreprocessor, raw_df: pd.DataFrame
) -> None:
    X, y, feature_names = preprocessor.prepare_features(raw_df, fit=True)
    assert X.shape[0] == len(raw_df)
    assert y is not None
    assert len(y) == len(raw_df)
    assert len(feature_names) == X.shape[1]


def test_prepare_features_target_binary(
    preprocessor: TelcoPreprocessor, raw_df: pd.DataFrame
) -> None:
    _, y, _ = preprocessor.prepare_features(raw_df, fit=True)
    assert y is not None
    assert set(y.unique()).issubset({0, 1})


def test_prepare_features_no_churn_column_in_x(
    preprocessor: TelcoPreprocessor, raw_df: pd.DataFrame
) -> None:
    X, _, _ = preprocessor.prepare_features(raw_df, fit=True)
    assert "Churn" not in X.columns


def test_is_fitted_after_prepare(
    preprocessor: TelcoPreprocessor, raw_df: pd.DataFrame
) -> None:
    preprocessor.prepare_features(raw_df, fit=True)
    assert preprocessor.is_fitted is True
