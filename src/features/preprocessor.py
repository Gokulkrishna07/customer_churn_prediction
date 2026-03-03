import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class TelcoPreprocessor:
    def __init__(self) -> None:
        self.scaler = StandardScaler()
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        missing_mask = df["TotalCharges"].isnull()
        df.loc[missing_mask, "TotalCharges"] = 0
        logger.info("Filled %d missing TotalCharges values", missing_mask.sum())
        if "customerID" in df.columns:
            df = df.drop("customerID", axis=1)
        df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
        logger.info("Data cleaned: shape=%s", df.shape)
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 72],
            labels=["0-1year", "1-2years", "2-4years", "4-6years"],
        )
        df["charges_ratio"] = df["TotalCharges"] / (df["MonthlyCharges"] + 1)
        df["avg_monthly_charges"] = df["TotalCharges"] / (df["tenure"] + 1)

        service_columns = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        df["total_services"] = sum(
            (df[col] == "Yes").astype(int)
            for col in service_columns
            if col in df.columns
        )
        df["has_internet"] = (df["InternetService"] != "No").astype(int)
        df["has_phone"] = (df["PhoneService"] == "Yes").astype(int)

        premium_services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        df["premium_services"] = sum(
            (df[col] == "Yes").astype(int)
            for col in premium_services
            if col in df.columns
        )
        df["streaming_services"] = sum(
            (df[col] == "Yes").astype(int)
            for col in ["StreamingTV", "StreamingMovies"]
            if col in df.columns
        )
        df["has_family"] = (
            (df["Partner"] == "Yes") | (df["Dependents"] == "Yes")
        ).astype(int)
        df["is_senior_with_family"] = (
            (df["SeniorCitizen"] == "Yes") & (df["has_family"] == 1)
        ).astype(int)
        df["auto_payment"] = df["PaymentMethod"].isin(
            ["Bank transfer (automatic)", "Credit card (automatic)"]
        ).astype(int)
        df["monthly_charges_per_service"] = df["MonthlyCharges"] / (df["total_services"] + 1)

        logger.info("Feature engineering complete: shape=%s", df.shape)
        return df

    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        target: pd.Series | None = None
        if "Churn" in df.columns:
            target = df["Churn"].copy()
            df = df.drop("Churn", axis=1)

        for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling", "SeniorCitizen"]:
            if col in df.columns:
                df[col] = (df[col] == "Yes").astype(int)

        for col in ["MultipleLines", "OnlineSecurity", "OnlineBackup",
                    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
            if col in df.columns:
                df[col] = df[col].replace(
                    {"No phone service": "No", "No internet service": "No"}
                )
                df[col] = (df[col] == "Yes").astype(int)

        ohe_cols = ["gender", "InternetService", "Contract", "PaymentMethod"]
        if "tenure_group" in df.columns:
            ohe_cols.append("tenure_group")
        for col in ohe_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)

        if target is not None:
            df["Churn"] = (target == "Yes").astype(int)

        logger.info("Encoding complete: shape=%s", df.shape)
        return df

    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        df = df.copy()
        target: pd.Series | None = None
        if "Churn" in df.columns:
            target = df["Churn"]
            df = df.drop("Churn", axis=1)

        scale_cols = [
            c for c in [
                "tenure", "MonthlyCharges", "TotalCharges",
                "charges_ratio", "avg_monthly_charges", "monthly_charges_per_service",
            ]
            if c in df.columns
        ]
        if fit:
            df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        else:
            df[scale_cols] = self.scaler.transform(df[scale_cols])

        if target is not None:
            df["Churn"] = target
        return df

    def prepare_features(
        self, df: pd.DataFrame, fit: bool = True
    ) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
        df = self.clean_data(df)
        df = self.engineer_features(df)
        df = self.encode_features(df, fit=fit)
        df = self.scale_features(df, fit=fit)

        y: pd.Series | None = None
        if "Churn" in df.columns:
            X = df.drop("Churn", axis=1)
            y = df["Churn"]
        else:
            X = df

        self.feature_names = list(X.columns)
        self.is_fitted = True
        logger.info("Preprocessing complete: features=%d", len(self.feature_names))
        return X, y, self.feature_names

    def save(self, path: str = "models/preprocessor.pkl") -> None:
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call prepare_features first.")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Preprocessor saved: path=%s", path)

    @classmethod
    def load(cls, path: str = "models/preprocessor.pkl") -> "TelcoPreprocessor":
        preprocessor: TelcoPreprocessor = joblib.load(path)
        logger.info("Preprocessor loaded: path=%s", path)
        return preprocessor
