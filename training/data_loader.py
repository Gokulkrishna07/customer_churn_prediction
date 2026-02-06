"""
Data loader for IBM Telco Customer Churn dataset.
Handles data loading and initial validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelcoDataLoader:
    """Loads and validates the Telco customer churn dataset."""
    
    def __init__(self, data_path: str = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to the raw CSV file
        """
        # Resolve path relative to project root (parent of training directory)
        if not Path(data_path).is_absolute():
            project_root = Path(__file__).parent.parent
            self.data_path = project_root / data_path
        else:
            self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from CSV.
        
        Returns:
            Loaded DataFrame
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        logger.info(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        
        logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
        logger.info(f"Columns: {list(self.df.columns)}")
        
        return self.df
    
    def get_data_info(self) -> dict:
        """
        Get comprehensive information about the dataset.
        
        Returns:
            Dictionary containing data statistics
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        info = {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'missing_percentage': (self.df.isnull().sum() / len(self.df) * 100).to_dict(),
            'duplicates': self.df.duplicated().sum(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Target variable distribution
        if 'Churn' in self.df.columns:
            info['churn_distribution'] = self.df['Churn'].value_counts().to_dict()
            info['churn_rate'] = (self.df['Churn'] == 'Yes').mean()
        
        return info
    
    def print_data_summary(self):
        """Print detailed data summary."""
        info = self.get_data_info()
        
        print("=" * 80)
        print("TELCO CUSTOMER CHURN DATASET SUMMARY")
        print("=" * 80)
        print(f"\nDataset Shape: {info['shape'][0]} rows × {info['shape'][1]} columns")
        print(f"Memory Usage: {info['memory_usage_mb']:.2f} MB")
        print(f"Duplicate Rows: {info['duplicates']}")
        
        print("\n" + "-" * 80)
        print("TARGET VARIABLE (Churn)")
        print("-" * 80)
        if 'churn_distribution' in info:
            for key, value in info['churn_distribution'].items():
                percentage = (value / info['shape'][0]) * 100
                print(f"{key}: {value:,} ({percentage:.2f}%)")
            print(f"\nChurn Rate: {info['churn_rate']:.2%}")
        
        print("\n" + "-" * 80)
        print("MISSING VALUES")
        print("-" * 80)
        missing_data = [(col, count, pct) 
                       for col, count, pct in zip(
                           info['missing_values'].keys(),
                           info['missing_values'].values(),
                           info['missing_percentage'].values()
                       ) if count > 0]
        
        if missing_data:
            for col, count, pct in missing_data:
                print(f"{col}: {count} ({pct:.2f}%)")
        else:
            print("No missing values found")
        
        print("\n" + "-" * 80)
        print("DATA TYPES")
        print("-" * 80)
        dtype_summary = {}
        for col, dtype in info['dtypes'].items():
            dtype_str = str(dtype)
            if dtype_str not in dtype_summary:
                dtype_summary[dtype_str] = []
            dtype_summary[dtype_str].append(col)
        
        for dtype, cols in dtype_summary.items():
            print(f"\n{dtype} ({len(cols)} columns):")
            for col in cols[:5]:  # Show first 5
                print(f"  - {col}")
            if len(cols) > 5:
                print(f"  ... and {len(cols) - 5} more")
        
        print("\n" + "=" * 80)
    
    def get_numerical_features(self) -> list:
        """Get list of numerical feature columns."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        return self.df.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_features(self) -> list:
        """Get list of categorical feature columns."""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        categorical = self.df.select_dtypes(include=['object']).columns.tolist()
        # Remove target variable if present
        if 'Churn' in categorical:
            categorical.remove('Churn')
        # Remove customer ID if present
        if 'customerID' in categorical:
            categorical.remove('customerID')
        
        return categorical


if __name__ == "__main__":
    # Test the data loader
    loader = TelcoDataLoader()
    df = loader.load_data()
    loader.print_data_summary()
    
    print(f"\nNumerical features: {loader.get_numerical_features()}")
    print(f"\nCategorical features: {loader.get_categorical_features()}")