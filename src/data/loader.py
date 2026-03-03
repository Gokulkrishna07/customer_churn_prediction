import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TelcoDataLoader:
    def __init__(self, data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv") -> None:
        if not Path(data_path).is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            self.data_path = project_root / data_path
        else:
            self.data_path = Path(data_path)
        self.df: pd.DataFrame | None = None

    def load_data(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info("Loaded data: shape=%s", self.df.shape)
        return self.df

    def get_data_info(self) -> dict[str, object]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        info: dict[str, object] = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "missing_values": self.df.isnull().sum().to_dict(),
            "duplicates": int(self.df.duplicated().sum()),
        }
        if "Churn" in self.df.columns:
            info["churn_rate"] = float((self.df["Churn"] == "Yes").mean())
        return info

    def get_numerical_features(self) -> list[str]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def get_categorical_features(self) -> list[str]:
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        categorical = self.df.select_dtypes(include=["object"]).columns.tolist()
        for col in ("Churn", "customerID"):
            if col in categorical:
                categorical.remove(col)
        return categorical
