"""
Training script with MLflow integration
"""
from data_loader import TelcoDataLoader
from preprocessing import TelcoPreprocessor
from model_trainer import ChurnModelTrainer

# MLflow server configuration
MLFLOW_TRACKING_URI = "http://103.49.125.28:5000"
EXPERIMENT_NAME = "churn-prediction"

# Load data
loader = TelcoDataLoader()
df = loader.load_data()

# Preprocess
preprocessor = TelcoPreprocessor()
X, y, feature_names = preprocessor.prepare_features(df)

# Initialize trainer with MLflow
trainer = ChurnModelTrainer(
    random_state=42,
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    experiment_name=EXPERIMENT_NAME
)

# Split data
X_train, X_test, y_train, y_test = trainer.split_data(X, y)

# Train with MLflow logging
trainer.train_all_models(
    X_train, X_test, y_train, y_test,
    imbalance_method='smote',
    cv=True,
    log_to_mlflow=True
)

# Display results
print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)
print(trainer.get_results_dataframe().to_string(index=False))

# Save best model locally
trainer.save_model()
trainer.save_all_results()
