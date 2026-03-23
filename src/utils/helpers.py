"""Utility helpers for data processing."""

import pandas as pd


def calculate_churn_rate(df: pd.DataFrame, target_col: str = "Churn") -> float:
    """Calculate the churn rate from a dataframe."""
    if target_col not in df.columns:
        return 0.0
    total = len(df)
    if total == 0:
        return 0.0
    churned = df[target_col].sum()
    return churned / total


def format_prediction_output(customer_id, probability, threshold=0.5):
    """Format a single prediction result."""
    label = "Churn" if probability >= threshold else "No Churn"
    return {
        "customer_id": customer_id,
        "churn_probability": round(probability, 4),
        "prediction": label,
        "risk_level": "high" if probability > 0.8 else "medium" if probability > 0.5 else "low",
    }
