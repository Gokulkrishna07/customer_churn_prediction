"""
Data preprocessing and feature engineering for Telco churn prediction.
Handles data cleaning, encoding, and feature creation.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
import joblib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TelcoPreprocessor:
    """
    Preprocesses Telco customer churn data.
    Implements data cleaning, encoding, and feature engineering.
    """
    
    def __init__(self):
        """Initialize preprocessor with encoding mappings."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data.
        
        Args:
            df: Raw dataframe
            
        Returns:
            Cleaned dataframe
        """
        df = df.copy()
        
        logger.info("Starting data cleaning...")
        
        # Fix TotalCharges - it has spaces for new customers
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # Fill missing TotalCharges with 0 (new customers with 0 tenure)
        mask = df['TotalCharges'].isnull()
        df.loc[mask, 'TotalCharges'] = 0
        logger.info(f"Filled {mask.sum()} missing TotalCharges values")
        
        # Remove customerID as it's not a feature
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
            logger.info("Removed customerID column")
        
        # Convert SeniorCitizen to categorical for consistency
        df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
        
        logger.info(f"Data cleaned. Final shape: {df.shape}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            df: Cleaned dataframe
            
        Returns:
            Dataframe with engineered features
        """
        df = df.copy()
        
        logger.info("Starting feature engineering...")
        
        # 1. Tenure-based features
        df['tenure_group'] = pd.cut(
            df['tenure'], 
            bins=[0, 12, 24, 48, 72],
            labels=['0-1year', '1-2years', '2-4years', '4-6years']
        )
        
        # 2. Charges-based features
        df['charges_ratio'] = df['TotalCharges'] / (df['MonthlyCharges'] + 1)
        df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        # 3. Service-based features
        service_columns = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        # Count total services
        df['total_services'] = 0
        for col in service_columns:
            if col in df.columns:
                # Count 'Yes' responses, treating 'No internet service' as 'No'
                df['total_services'] += (df[col] == 'Yes').astype(int)
        
        # Internet service flag
        df['has_internet'] = (df['InternetService'] != 'No').astype(int)
        
        # Phone service flag
        df['has_phone'] = (df['PhoneService'] == 'Yes').astype(int)
        
        # Premium services (security, backup, protection, support)
        premium_services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
        df['premium_services'] = 0
        for col in premium_services:
            if col in df.columns:
                df['premium_services'] += (df[col] == 'Yes').astype(int)
        
        # Streaming services
        df['streaming_services'] = 0
        for col in ['StreamingTV', 'StreamingMovies']:
            if col in df.columns:
                df['streaming_services'] += (df[col] == 'Yes').astype(int)
        
        # 4. Customer profile features
        df['has_family'] = ((df['Partner'] == 'Yes') | (df['Dependents'] == 'Yes')).astype(int)
        df['is_senior_with_family'] = ((df['SeniorCitizen'] == 'Yes') & (df['has_family'] == 1)).astype(int)
        
        # 5. Contract and payment features
        df['auto_payment'] = df['PaymentMethod'].isin([
            'Bank transfer (automatic)', 
            'Credit card (automatic)'
        ]).astype(int)
        
        # 6. Value-based features
        df['monthly_charges_per_service'] = df['MonthlyCharges'] / (df['total_services'] + 1)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        logger.info(f"Created {df.shape[1] - 20} new features")
        
        return df
    
    def encode_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features.
        
        Args:
            df: Dataframe with engineered features
            fit: Whether to fit encoders or use existing ones
            
        Returns:
            Dataframe with encoded features
        """
        df = df.copy()
        
        logger.info("Starting feature encoding...")
        
        # Separate target variable
        if 'Churn' in df.columns:
            target = df['Churn'].copy()
            df = df.drop('Churn', axis=1)
        else:
            target = None
        
        # Binary encoding for Yes/No columns
        binary_columns = [
            'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'SeniorCitizen'
        ]
        
        for col in binary_columns:
            if col in df.columns:
                df[col] = (df[col] == 'Yes').astype(int)
        
        # Handle columns with 'No internet service' or 'No phone service'
        service_columns = [
            'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        
        for col in service_columns:
            if col in df.columns:
                df[col] = df[col].replace({
                    'No phone service': 'No',
                    'No internet service': 'No'
                })
                df[col] = (df[col] == 'Yes').astype(int)
        
        # One-hot encoding for multi-category features
        categorical_columns = ['gender', 'InternetService', 'Contract', 'PaymentMethod']
        
        # Add tenure_group if it exists
        if 'tenure_group' in df.columns:
            categorical_columns.append('tenure_group')
        
        for col in categorical_columns:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)
        
        # Add target back if it existed
        if target is not None:
            df['Churn'] = (target == 'Yes').astype(int)
        
        logger.info(f"Encoding completed. Final shape: {df.shape}")
        
        return df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Dataframe with encoded features
            fit: Whether to fit scaler or use existing one
            
        Returns:
            Dataframe with scaled features
        """
        df = df.copy()
        
        # Separate target if present
        if 'Churn' in df.columns:
            target = df['Churn']
            df = df.drop('Churn', axis=1)
        else:
            target = None
        
        # Features to scale (continuous numerical features)
        scale_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 
                         'charges_ratio', 'avg_monthly_charges', 
                         'monthly_charges_per_service']
        
        scale_features = [f for f in scale_features if f in df.columns]
        
        if fit:
            df[scale_features] = self.scaler.fit_transform(df[scale_features])
            logger.info(f"Fitted and transformed {len(scale_features)} numerical features")
        else:
            df[scale_features] = self.scaler.transform(df[scale_features])
            logger.info(f"Transformed {len(scale_features)} numerical features")
        
        # Add target back
        if target is not None:
            df['Churn'] = target
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit: bool = True) -> tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Raw dataframe
            fit: Whether to fit preprocessors
            
        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Starting complete preprocessing pipeline...")
        
        # Step 1: Clean data
        df = self.clean_data(df)
        
        # Step 2: Engineer features
        df = self.engineer_features(df)
        
        # Step 3: Encode features
        df = self.encode_features(df, fit=fit)
        
        # Step 4: Scale features
        df = self.scale_features(df, fit=fit)
        
        # Step 5: Separate features and target
        if 'Churn' in df.columns:
            X = df.drop('Churn', axis=1)
            y = df['Churn']
        else:
            X = df
            y = None
        
        # Store feature names
        self.feature_names = list(X.columns)
        self.is_fitted = True
        
        logger.info(f"Preprocessing complete. Feature shape: {X.shape}")
        logger.info(f"Total features: {len(self.feature_names)}")
        
        if y is not None:
            logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, self.feature_names
    
    def save(self, path: str = "models/preprocessor.pkl"):
        """
        Save the fitted preprocessor.
        
        Args:
            path: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call prepare_features first.")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self, path)
        logger.info(f"Preprocessor saved to {path}")
    
    @classmethod
    def load(cls, path: str = "models/preprocessor.pkl"):
        """
        Load a fitted preprocessor.
        
        Args:
            path: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor
        """
        preprocessor = joblib.load(path)
        logger.info(f"Preprocessor loaded from {path}")
        return preprocessor


if __name__ == "__main__":
    # Test the preprocessor
    from data_loader import TelcoDataLoader
    
    loader = TelcoDataLoader()
    df = loader.load_data()
    
    preprocessor = TelcoPreprocessor()
    X, y, feature_names = preprocessor.prepare_features(df)
    
    print("\n" + "=" * 80)
    print("PREPROCESSING SUMMARY")
    print("=" * 80)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Total features created: {len(feature_names)}")
    print(f"\nFirst 10 features: {feature_names[:10]}")
    print(f"\nTarget distribution:\n{y.value_counts()}")
    print(f"Churn rate: {y.mean():.2%}")