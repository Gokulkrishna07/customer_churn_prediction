"""
Final production-ready model packaging.
Creates deployable model with optimal threshold and preprocessor.
"""

import joblib
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionChurnModel:
    """
    Production-ready churn prediction model.
    Wraps model, preprocessor, and optimal threshold.
    """
    
    def __init__(self, model, preprocessor, threshold: float = 0.5):
        """
        Initialize production model.
        
        Args:
            model: Trained model
            preprocessor: Fitted preprocessor
            threshold: Optimal prediction threshold
        """
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold
        self.metadata = {
            'created_at': datetime.now().isoformat(),
            'threshold': threshold,
            'model_type': type(model).__name__,
            'feature_names': preprocessor.feature_names
        }
    
    def predict(self, X):
        """
        Make predictions with optimal threshold.
        
        Args:
            X: Input features (can be raw DataFrame or preprocessed array)
            
        Returns:
            Binary predictions
        """
        # Get probabilities
        proba = self.predict_proba(X)
        
        # Apply optimal threshold
        predictions = (proba[:, 1] >= self.threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Probability array [prob_no_churn, prob_churn]
        """
        return self.model.predict_proba(X)
    
    def predict_single(self, customer_data: dict) -> dict:
        """
        Predict for a single customer.
        
        Args:
            customer_data: Dictionary of customer features
            
        Returns:
            Dictionary with prediction and probability
        """
        import pandas as pd
        
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        
        # Preprocess using the transform method to maintain feature alignment
        df_cleaned = self.preprocessor.clean_data(df)
        df_engineered = self.preprocessor.engineer_features(df_cleaned)
        df_encoded = self.preprocessor.encode_features(df_engineered, fit=False)
        df_scaled = self.preprocessor.scale_features(df_encoded, fit=False)
        
        # Remove target if present
        if 'Churn' in df_scaled.columns:
            X = df_scaled.drop('Churn', axis=1)
        else:
            X = df_scaled
        
        # Ensure all expected features are present
        missing_features = set(self.preprocessor.feature_names) - set(X.columns)
        if missing_features:
            for feature in missing_features:
                X[feature] = 0
        
        # Reorder columns to match training
        X = X[self.preprocessor.feature_names]
        
        # Predict
        proba = self.predict_proba(X)[0]
        prediction = int(self.predict(X)[0])
        
        return {
            'churn_prediction': prediction,
            'churn_probability': float(proba[1]),
            'no_churn_probability': float(proba[0]),
            'confidence': float(abs(proba[1] - self.threshold) / (1 - self.threshold))
        }
    
    def get_feature_importance(self, top_n: int = 10) -> dict:
        """
        Get top feature importances.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            Dictionary of feature: importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        return {
            self.preprocessor.feature_names[i]: float(importances[i])
            for i in indices
        }
    
    def save(self, path: str = "models/production"):
        """
        Save production model package.
        
        Args:
            path: Directory to save model
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save complete package
        package_path = Path(path) / "churn_model_production.pkl"
        joblib.dump(self, package_path)
        logger.info(f"Production model saved to {package_path}")
        
        # Save metadata
        metadata_path = Path(path) / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=4)
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Save feature importance
        importance_path = Path(path) / "feature_importance.json"
        with open(importance_path, 'w') as f:
            json.dump(self.get_feature_importance(20), f, indent=4)
        logger.info(f"Feature importance saved to {importance_path}")
    
    @classmethod
    def load(cls, path: str = "models/production/churn_model_production.pkl"):
        """
        Load production model.
        
        Args:
            path: Path to model file
            
        Returns:
            Loaded ProductionChurnModel
        """
        model = joblib.load(path)
        logger.info(f"Production model loaded from {path}")
        return model


def create_production_model(
    model_path: str = "models/tuned/random_forest_tuned.pkl",
    preprocessor_path: str = "models/preprocessor.pkl",
    optimal_threshold: float = 0.38
):
    """
    Create production model from trained artifacts.
    
    Args:
        model_path: Path to trained model
        preprocessor_path: Path to fitted preprocessor
        optimal_threshold: Optimal prediction threshold
        
    Returns:
        ProductionChurnModel instance
    """
    logger.info("Creating production model package...")
    
    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
    except FileNotFoundError:
        # Fallback to default model
        model_path = "models/random_forest.pkl"
        model = joblib.load(model_path)
        logger.info(f"Loaded fallback model from {model_path}")
    
    # Load preprocessor
    preprocessor = joblib.load(preprocessor_path)
    logger.info(f"Loaded preprocessor from {preprocessor_path}")
    
    # Create production model
    prod_model = ProductionChurnModel(model, preprocessor, optimal_threshold)
    
    logger.info("Production model created successfully")
    logger.info(f"  Model type: {type(model).__name__}")
    logger.info(f"  Optimal threshold: {optimal_threshold}")
    logger.info(f"  Number of features: {len(preprocessor.feature_names)}")
    
    return prod_model


def test_production_model(model: ProductionChurnModel):
    """
    Test production model with sample data.
    
    Args:
        model: ProductionChurnModel instance
    """
    logger.info("\nTesting production model with sample customer...")
    
    # Sample customer (high churn risk)
    sample_customer = {
        'gender': 'Male',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 2,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'Yes',
        'StreamingMovies': 'Yes',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 85.5,
        'TotalCharges': 171.0
    }
    
    result = model.predict_single(sample_customer)
    
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTION")
    print("=" * 80)
    print(f"Customer Profile:")
    print(f"  Tenure: {sample_customer['tenure']} months")
    print(f"  Contract: {sample_customer['Contract']}")
    print(f"  Monthly Charges: ${sample_customer['MonthlyCharges']}")
    print(f"  Internet: {sample_customer['InternetService']}")
    print(f"\nPrediction Results:")
    print(f"  Churn Prediction: {'YES' if result['churn_prediction'] == 1 else 'NO'}")
    print(f"  Churn Probability: {result['churn_probability']:.2%}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print("=" * 80)


if __name__ == "__main__":
    # Create production model
    prod_model = create_production_model(
        model_path="models/tuned/random_forest_tuned.pkl",
        preprocessor_path="models/preprocessor.pkl",
        optimal_threshold=0.38  # From threshold analysis
    )
    
    # Test model
    test_production_model(prod_model)
    
    # Save production model
    prod_model.save()
    
    # Show feature importance
    print("\n" + "=" * 80)
    print("TOP 10 MOST IMPORTANT FEATURES")
    print("=" * 80)
    for i, (feature, importance) in enumerate(prod_model.get_feature_importance(10).items(), 1):
        print(f"{i:2}. {feature:<40} {importance:.6f}")
    print("=" * 80)
    
    logger.info("\n✓ Production model ready for deployment!")