"""
Model training pipeline for Telco churn prediction.
Implements multiple algorithms with hyperparameter tuning.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import logging
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    """
    Trains and evaluates churn prediction models.
    Supports multiple algorithms with class imbalance handling.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize trainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def get_models(self) -> dict:
        """
        Get dictionary of models to train.
        
        Returns:
            Dictionary of model instances
        """
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=self.random_state,
                class_weight='balanced',
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            ),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                eval_metric='logloss',
                scale_pos_weight=1
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state,
                class_weight='balanced',
                verbose=-1
            )
        }
        
        return models
    
    def split_data(self, X, y, test_size: float = 0.2):
        """
        Split data into train and test sets.
        
        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {X_train.shape[0]} samples")
        logger.info(f"  Test set: {X_test.shape[0]} samples")
        logger.info(f"  Train churn rate: {y_train.mean():.2%}")
        logger.info(f"  Test churn rate: {y_test.mean():.2%}")
        
        return X_train, X_test, y_train, y_test
    
    def handle_imbalance(self, X_train, y_train, method: str = 'smote'):
        """
        Handle class imbalance.
        
        Args:
            X_train: Training features
            y_train: Training target
            method: Method to use ('smote', 'undersample', 'none')
            
        Returns:
            Resampled X_train, y_train
        """
        if method == 'none':
            logger.info("No resampling applied")
            return X_train, y_train
        
        logger.info(f"Applying {method} for class imbalance...")
        logger.info(f"Before resampling: {np.bincount(y_train)}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.random_state)
        elif method == 'undersample':
            sampler = RandomUnderSampler(random_state=self.random_state)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        
        logger.info(f"After resampling: {np.bincount(y_resampled)}")
        
        return X_resampled, y_resampled
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> dict:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        # Log results
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model, X, y, cv: int = 5) -> dict:
        """
        Perform cross-validation.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Dictionary of CV scores
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scores = {
            'accuracy': cross_val_score(model, X, y, cv=skf, scoring='accuracy'),
            'precision': cross_val_score(model, X, y, cv=skf, scoring='precision'),
            'recall': cross_val_score(model, X, y, cv=skf, scoring='recall'),
            'f1': cross_val_score(model, X, y, cv=skf, scoring='f1'),
            'roc_auc': cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        }
        
        cv_results = {
            f'{metric}_mean': scores[metric].mean()
            for metric in scores
        }
        cv_results.update({
            f'{metric}_std': scores[metric].std()
            for metric in scores
        })
        
        return cv_results
    
    def train_all_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        imbalance_method: str = 'smote',
        cv: bool = True
    ):
        """
        Train all models and compare results.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            imbalance_method: Method to handle imbalance
            cv: Whether to perform cross-validation
        """
        logger.info("=" * 80)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 80)
        
        # Handle class imbalance
        X_train_resampled, y_train_resampled = self.handle_imbalance(
            X_train, y_train, method=imbalance_method
        )
        
        # Get models
        models = self.get_models()
        
        # Train each model
        for model_name, model in models.items():
            logger.info(f"\nTraining {model_name}...")
            
            # Train model
            model.fit(X_train_resampled, y_train_resampled)
            self.models[model_name] = model
            
            # Evaluate on test set
            metrics = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Cross-validation
            if cv:
                logger.info(f"Performing cross-validation for {model_name}...")
                cv_results = self.cross_validate_model(model, X_train, y_train)
                metrics['cv_results'] = cv_results
                
                logger.info(f"  CV ROC-AUC: {cv_results['roc_auc_mean']:.4f} (+/- {cv_results['roc_auc_std']:.4f})")
                logger.info(f"  CV F1:      {cv_results['f1_mean']:.4f} (+/- {cv_results['f1_std']:.4f})")
            
            self.results[model_name] = metrics
        
        # Determine best model
        self._select_best_model()
        
        logger.info("\n" + "=" * 80)
        logger.info(f"BEST MODEL: {self.best_model_name}")
        logger.info("=" * 80)
    
    def _select_best_model(self):
        """Select best model based on F1-score."""
        best_f1 = 0
        
        for model_name, metrics in self.results.items():
            if metrics['f1_score'] > best_f1:
                best_f1 = metrics['f1_score']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        logger.info(f"\nBest model selected: {self.best_model_name} (F1={best_f1:.4f})")
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get results as a DataFrame for comparison.
        
        Returns:
            DataFrame with all model results
        """
        results_list = []
        
        for model_name, metrics in self.results.items():
            result = {
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'ROC-AUC': metrics['roc_auc']
            }
            
            if 'cv_results' in metrics:
                result['CV_F1_Mean'] = metrics['cv_results']['f1_mean']
                result['CV_ROC_AUC_Mean'] = metrics['cv_results']['roc_auc_mean']
            
            results_list.append(result)
        
        df_results = pd.DataFrame(results_list)
        df_results = df_results.sort_values('F1-Score', ascending=False)
        
        return df_results
    
    def save_model(self, model_name: str = None, path: str = "models"):
        """
        Save trained model.
        
        Args:
            model_name: Name of model to save (None = best model)
            path: Directory to save model
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = Path(path) / f"{model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save results
        results_path = Path(path) / f"{model_name.replace(' ', '_').lower()}_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results[model_name], f, indent=4)
        logger.info(f"Results saved to {results_path}")
    
    def save_all_results(self, path: str = "results"):
        """
        Save all training results.
        
        Args:
            path: Directory to save results
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save results DataFrame
        df_results = self.get_results_dataframe()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(path) / f"training_results_{timestamp}.csv"
        df_results.to_csv(csv_path, index=False)
        logger.info(f"Results DataFrame saved to {csv_path}")
        
        # Save detailed results
        json_path = Path(path) / f"detailed_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        logger.info(f"Detailed results saved to {json_path}")


if __name__ == "__main__":
    # Test the trainer
    from data_loader import TelcoDataLoader
    from preprocessing import TelcoPreprocessor
    
    # Load and preprocess data
    loader = TelcoDataLoader()
    df = loader.load_data()
    
    preprocessor = TelcoPreprocessor()
    X, y, feature_names = preprocessor.prepare_features(df)
    
    # Initialize trainer
    trainer = ChurnModelTrainer(random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Train all models
    trainer.train_all_models(
        X_train, X_test, y_train, y_test,
        imbalance_method='smote',
        cv=True
    )
    
    # Display results
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(trainer.get_results_dataframe().to_string(index=False))
    
    # Save best model
    trainer.save_model()
    trainer.save_all_results()