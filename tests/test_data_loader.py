import pandas as pd
import pytest

from src.data.loader import TelcoDataLoader


@pytest.fixture()
def loader() -> TelcoDataLoader:
    return TelcoDataLoader()


def test_load_data_returns_dataframe(loader: TelcoDataLoader) -> None:
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_has_expected_columns(loader: TelcoDataLoader) -> None:
    df = loader.load_data()
    assert "Churn" in df.columns
    assert "customerID" in df.columns
    assert "tenure" in df.columns
    assert "MonthlyCharges" in df.columns


def test_load_data_churn_column_values(loader: TelcoDataLoader) -> None:
    df = loader.load_data()
    assert set(df["Churn"].unique()).issubset({"Yes", "No"})


def test_get_data_info_requires_load(loader: TelcoDataLoader) -> None:
    with pytest.raises(ValueError, match="not loaded"):
        loader.get_data_info()


def test_get_data_info_returns_churn_rate(loader: TelcoDataLoader) -> None:
    loader.load_data()
    info = loader.get_data_info()
    assert "churn_rate" in info
    assert 0.0 < float(info["churn_rate"]) < 1.0  # type: ignore[arg-type]


def test_file_not_found_raises() -> None:
    loader = TelcoDataLoader(data_path="data/raw/nonexistent.csv")
    with pytest.raises(FileNotFoundError):
        loader.load_data()
