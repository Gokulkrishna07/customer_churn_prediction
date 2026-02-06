"""
Hyperparameter tuning for churn prediction models.
Uses GridSearchCV and RandomizedSearchCV for optimization.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import joblib
import logging
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import randint, uniform

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Performs hyperparameter tuning for ML models.
    Supports both GridSearch and RandomizedSearch.
    """
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        """
        Initialize tuner.
        
        Args:
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.best_params = {}
        self.best_models = {}
        self.cv_results = {}
        
    def get_param_grid(self, model_name: str, search_type: str = 'random') -> dict:
        """
        Get parameter grid for a model.
        
        Args:
            model_name: Name of the model
            search_type: 'grid' for GridSearch, 'random' for RandomizedSearch
            
        Returns:
            Parameter grid dictionary
        """
        if model_name == 'Random Forest':
            if search_type == 'grid':
                return {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2'],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
            else:  # random
                return {
                    'n_estimators': randint(100, 500),
                    'max_depth': [10, 15, 20, 25, 30, None],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None],
                    'class_weight': ['balanced', 'balanced_subsample']
                }
        
        elif model_name == 'Gradient Boosting':
            if search_type == 'grid':
                return {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'subsample': [0.8, 0.9, 1.0]
                }
            else:  # random
                return {
                    'n_estimators': randint(100, 500),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 10),
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'subsample': uniform(0.7, 0.3)
                }
        
        elif model_name == 'XGBoost':
            if search_type == 'grid':
                return {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'gamma': [0, 0.1, 0.2]
                }
            else:  # random
                return {
                    'n_estimators': randint(100, 500),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(3, 10),
                    'min_child_weight': randint(1, 10),
                    'subsample': uniform(0.7, 0.3),
                    'colsample_bytree': uniform(0.7, 0.3),
                    'gamma': uniform(0, 0.3)
                }
        
        elif model_name == 'LightGBM':
            if search_type == 'grid':
                return {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [5, 10, 15],
                    'num_leaves': [31, 50, 70],
                    'min_child_samples': [10, 20, 30],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0]
                }
            else:  # random
                return {
                    'n_estimators': randint(100, 500),
                    'learning_rate': uniform(0.01, 0.2),
                    'max_depth': randint(5, 20),
                    'num_leaves': randint(20, 100),
                    'min_child_samples': randint(10, 50),
                    'subsample': uniform(0.7, 0.3),
                    'colsample_bytree': uniform(0.7, 0.3)
                }
        
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def get_base_model(self, model_name: str):
        """
        Get base model instance.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Base model instance
        """
        if model_name == 'Random Forest':
            return RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
        elif model_name == 'Gradient Boosting':
            return GradientBoostingClassifier(
                random_state=self.random_state
            )
        elif model_name == 'XGBoost':
            return XGBClassifier(
                random_state=self.random_state,
                eval_metric='logloss',
                n_jobs=self.n_jobs
            )
        elif model_name == 'LightGBM':
            return LGBMClassifier(
                random_state=self.random_state,
                verbose=-1,
                n_jobs=self.n_jobs
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def tune_model(
        self,
        model_name: str,
        X_train,
        y_train,
        search_type: str = 'random',
        n_iter: int = 50,
        cv: int = 5,
        scoring: str = 'f1'
    ):
        """
        Tune hyperparameters for a model.
        
        Args:
            model_name: Name of model to tune
            X_train: Training features
            y_train: Training target
            search_type: 'grid' or 'random'
            n_iter: Number of iterations for random search
            cv: Number of CV folds
            scoring: Scoring metric
            
        Returns:
            Best model and parameters
        """
        logger.info("=" * 80)
        logger.info(f"TUNING {model_name.upper()} - {search_type.upper()} SEARCH")
        logger.info("=" * 80)
        
        # Get base model and parameter grid
        base_model = self.get_base_model(model_name)
        param_grid = self.get_param_grid(model_name, search_type)
        
        # Setup cross-validation
        cv_strategy = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )
        
        # Setup scorer
        if scoring == 'f1':
            scorer = make_scorer(f1_score)
        elif scoring == 'roc_auc':
            scorer = make_scorer(roc_auc_score, needs_proba=True)
        else:
            scorer = scoring
        
        # Perform search
        if search_type == 'grid':
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=cv_strategy,
                scoring=scorer,
                n_jobs=self.n_jobs,
                verbose=2,
                return_train_score=True
            )
            logger.info(f"Starting GridSearchCV with {len(param_grid)} parameters...")
        else:  # random
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=n_iter,
                cv=cv_strategy,
                scoring=scorer,
                n_jobs=self.n_jobs,
                verbose=2,
                random_state=self.random_state,
                return_train_score=True
            )
            logger.info(f"Starting RandomizedSearchCV with {n_iter} iterations...")
        
        # Fit search
        search.fit(X_train, y_train)
        
        # Store results
        self.best_params[model_name] = search.best_params_
        self.best_models[model_name] = search.best_estimator_
        
        # Store CV results
        cv_results = pd.DataFrame(search.cv_results_)
        self.cv_results[model_name] = cv_results
        
        # Log results
        logger.info(f"\nBest parameters for {model_name}:")
        for param, value in search.best_params_.items():
            logger.info(f"  {param}: {value}")
        
        logger.info(f"\nBest CV score: {search.best_score_:.4f}")
        logger.info(f"Best estimator: {search.best_estimator_}")
        
        return search.best_estimator_, search.best_params_
    
    def tune_top_models(
        self,
        X_train,
        y_train,
        models: list = None,
        search_type: str = 'random',
        n_iter: int = 50
    ):
        """
        Tune multiple models.
        
        Args:
            X_train: Training features
            y_train: Training target
            models: List of model names to tune
            search_type: 'grid' or 'random'
            n_iter: Number of iterations for random search
        """
        if models is None:
            models = ['Random Forest', 'XGBoost', 'LightGBM']
        
        logger.info(f"\nTuning {len(models)} models: {models}")
        
        for model_name in models:
            self.tune_model(
                model_name,
                X_train,
                y_train,
                search_type=search_type,
                n_iter=n_iter
            )
        
        logger.info("\n" + "=" * 80)
        logger.info("HYPERPARAMETER TUNING COMPLETED")
        logger.info("=" * 80)
    
    def save_results(self, path: str = "models/tuned"):
        """
        Save tuned models and results.
        
        Args:
            path: Directory to save results
        """
        Path(path).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save each tuned model
        for model_name, model in self.best_models.items():
            model_filename = f"{model_name.replace(' ', '_').lower()}_tuned.pkl"
            model_path = Path(path) / model_filename
            joblib.dump(model, model_path)
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save best parameters
        params_path = Path(path) / f"best_parameters_{timestamp}.json"
        with open(params_path, 'w') as f:
            json.dump(self.best_params, f, indent=4)
        logger.info(f"Saved best parameters to {params_path}")
        
        # Save CV results
        for model_name, cv_results in self.cv_results.items():
            cv_filename = f"{model_name.replace(' ', '_').lower()}_cv_results.csv"
            cv_path = Path(path) / cv_filename
            cv_results.to_csv(cv_path, index=False)
            logger.info(f"Saved CV results for {model_name} to {cv_path}")
    
    def compare_with_baseline(self, baseline_results: dict):
        """
        Compare tuned models with baseline.
        
        Args:
            baseline_results: Dictionary of baseline model results
        """
        print("\n" + "=" * 80)
        print("BASELINE VS TUNED MODEL COMPARISON")
        print("=" * 80)
        
        for model_name in self.best_models.keys():
            if model_name in baseline_results:
                baseline_f1 = baseline_results[model_name]['f1_score']
                baseline_auc = baseline_results[model_name]['roc_auc']
                
                print(f"\n{model_name}:")
                print(f"  Baseline F1:  {baseline_f1:.4f}")
                print(f"  Baseline AUC: {baseline_auc:.4f}")
                print(f"  (Tuned model needs to be evaluated on test set)")


if __name__ == "__main__":
    # Load and prepare data
    from data_loader import TelcoDataLoader
    from preprocessing import TelcoPreprocessor
    from model_trainer import ChurnModelTrainer
    
    logger.info("Loading and preprocessing data...")
    loader = TelcoDataLoader()
    df = loader.load_data()
    
    preprocessor = TelcoPreprocessor()
    X, y, feature_names = preprocessor.prepare_features(df)
    
    # Split data
    trainer = ChurnModelTrainer(random_state=42)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Handle imbalance
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    logger.info(f"Training set after SMOTE: {X_train_resampled.shape}")
    
    # Initialize tuner
    tuner = HyperparameterTuner(random_state=42)
    
    # Tune top models
    # Note: Start with Random Forest only to save time
    # You can add more models later
    tuner.tune_top_models(
        X_train_resampled,
        y_train_resampled,
        models=['Random Forest'],
        search_type='random',
        n_iter=30  # Reduced for faster execution
    )
    
    # Save results
    tuner.save_results()
    
    logger.info("\nHyperparameter tuning completed!")
    logger.info("Evaluate the tuned model on test set to see improvement.")