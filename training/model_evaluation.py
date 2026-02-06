"""
Comprehensive model evaluation with visualizations.
Generates evaluation reports and performance plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ModelEvaluator:
    """
    Comprehensive model evaluation with visualizations.
    Provides detailed performance analysis and reporting.
    """
    
    def __init__(self, model, model_name: str = "Model"):
        """
        Initialize evaluator.
        
        Args:
            model: Trained model
            model_name: Name of the model
        """
        self.model = model
        self.model_name = model_name
        self.metrics = {}
        
    def evaluate(self, X_test, y_test) -> dict:
        """
        Evaluate model on test set.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {self.model_name}...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'avg_precision': average_precision_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, 
                                                          target_names=['No Churn', 'Churn'])
        }
        
        # Calculate additional metrics
        tn, fp, fn, tp = self.metrics['confusion_matrix'].ravel()
        self.metrics['true_negatives'] = tn
        self.metrics['false_positives'] = fp
        self.metrics['false_negatives'] = fn
        self.metrics['true_positives'] = tp
        self.metrics['specificity'] = tn / (tn + fp)
        
        return self.metrics
    
    def print_evaluation_report(self):
        """Print detailed evaluation report."""
        print("\n" + "=" * 80)
        print(f"EVALUATION REPORT: {self.model_name}")
        print("=" * 80)
        
        print("\n1. OVERALL METRICS")
        print("-" * 80)
        print(f"Accuracy:           {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)")
        print(f"Precision:          {self.metrics['precision']:.4f} ({self.metrics['precision']*100:.2f}%)")
        print(f"Recall (Sensitivity): {self.metrics['recall']:.4f} ({self.metrics['recall']*100:.2f}%)")
        print(f"Specificity:        {self.metrics['specificity']:.4f} ({self.metrics['specificity']*100:.2f}%)")
        print(f"F1-Score:           {self.metrics['f1_score']:.4f}")
        print(f"ROC-AUC:            {self.metrics['roc_auc']:.4f}")
        print(f"Average Precision:  {self.metrics['avg_precision']:.4f}")
        
        print("\n2. CONFUSION MATRIX")
        print("-" * 80)
        cm = self.metrics['confusion_matrix']
        print(f"                    Predicted: No Churn    Predicted: Churn")
        print(f"Actual: No Churn    {cm[0,0]:>15}    {cm[0,1]:>15}")
        print(f"Actual: Churn       {cm[1,0]:>15}    {cm[1,1]:>15}")
        
        print("\n3. CONFUSION MATRIX BREAKDOWN")
        print("-" * 80)
        print(f"True Negatives (TN):  {self.metrics['true_negatives']:>6} (Correctly predicted No Churn)")
        print(f"False Positives (FP): {self.metrics['false_positives']:>6} (Incorrectly predicted Churn)")
        print(f"False Negatives (FN): {self.metrics['false_negatives']:>6} (Missed Churn cases)")
        print(f"True Positives (TP):  {self.metrics['true_positives']:>6} (Correctly predicted Churn)")
        
        print("\n4. DETAILED CLASSIFICATION REPORT")
        print("-" * 80)
        print(self.metrics['classification_report'])
        
        print("\n5. BUSINESS INTERPRETATION")
        print("-" * 80)
        total = len(self.metrics['confusion_matrix'].ravel())
        print(f"• Out of every 100 customers predicted to churn:")
        print(f"  - {self.metrics['precision']*100:.1f} actually churn (Precision)")
        print(f"• Out of every 100 customers who actually churn:")
        print(f"  - {self.metrics['recall']*100:.1f} are correctly identified (Recall)")
        print(f"• Overall, {self.metrics['accuracy']*100:.1f}% of predictions are correct")
        
        print("\n" + "=" * 80)
    
    def plot_confusion_matrix(self, save_path: str = None):
        """
        Plot confusion matrix heatmap.
        
        Args:
            save_path: Path to save plot (optional)
        """
        cm = self.metrics['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Churn', 'Churn'],
                    yticklabels=['No Churn', 'Churn'],
                    cbar_kws={'label': 'Count'})
        
        plt.title(f'Confusion Matrix - {self.model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('Actual', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12, fontweight='bold')
        
        # Add percentages
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                        ha='center', va='center', fontsize=10, color='red')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, X_test, y_test, save_path: str = None):
        """
        Plot ROC curve.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save plot (optional)
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = self.metrics['roc_auc']
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title(f'ROC Curve - {self.model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curve(self, X_test, y_test, save_path: str = None):
        """
        Plot Precision-Recall curve.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save plot (optional)
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        avg_precision = self.metrics['avg_precision']
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.4f})')
        plt.axhline(y=y_test.mean(), color='navy', linestyle='--', lw=2,
                   label=f'Baseline (No Skill = {y_test.mean():.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title(f'Precision-Recall Curve - {self.model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, feature_names: list, top_n: int = 20, 
                               save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save plot (optional)
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning(f"{self.model_name} does not have feature_importances_")
            return
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), importances[indices], color='skyblue', edgecolor='black')
        plt.yticks(range(top_n), [feature_names[i] for i in indices])
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Feature Importances - {self.model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
        # Print top features
        print("\n" + "=" * 80)
        print(f"TOP {top_n} MOST IMPORTANT FEATURES")
        print("=" * 80)
        for i, idx in enumerate(indices, 1):
            print(f"{i:2}. {feature_names[idx]:<40} {importances[idx]:.6f}")
        print("=" * 80)
    
    def plot_threshold_analysis(self, X_test, y_test, save_path: str = None):
        """
        Plot metrics vs threshold to find optimal threshold.
        
        Args:
            X_test: Test features
            y_test: Test labels
            save_path: Path to save plot (optional)
        """
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.02)
        
        precisions = []
        recalls = []
        f1_scores = []
        accuracies = []
        
        for threshold in thresholds:
            y_pred_threshold = (y_pred_proba >= threshold).astype(int)
            precisions.append(precision_score(y_test, y_pred_threshold))
            recalls.append(recall_score(y_test, y_pred_threshold))
            f1_scores.append(f1_score(y_test, y_pred_threshold))
            accuracies.append(accuracy_score(y_test, y_pred_threshold))
        
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1-Score', linewidth=2)
        plt.plot(thresholds, accuracies, label='Accuracy', linewidth=2)
        
        # Mark default threshold (0.5)
        plt.axvline(x=0.5, color='red', linestyle='--', linewidth=2, 
                   label='Default Threshold (0.5)')
        
        # Mark optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        plt.axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2,
                   label=f'Optimal F1 Threshold ({optimal_threshold:.2f})')
        
        plt.xlabel('Threshold', fontsize=12, fontweight='bold')
        plt.ylabel('Score', fontsize=12, fontweight='bold')
        plt.title(f'Metrics vs Threshold - {self.model_name}', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Threshold analysis plot saved to {save_path}")
        
        plt.show()
        
        print(f"\nOptimal threshold for F1-Score: {optimal_threshold:.2f}")
        print(f"F1-Score at optimal threshold: {f1_scores[optimal_idx]:.4f}")
    
    def generate_evaluation_report(self, X_test, y_test, feature_names: list, 
                                   output_dir: str = "results/evaluation"):
        """
        Generate complete evaluation report with all visualizations.
        
        Args:
            X_test: Test features
            y_test: Test labels
            feature_names: List of feature names
            output_dir: Directory to save results
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(output_dir) / f"{self.model_name.replace(' ', '_')}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating evaluation report in {model_dir}")
        
        # Evaluate model
        self.evaluate(X_test, y_test)
        
        # Print report
        self.print_evaluation_report()
        
        # Generate all plots
        self.plot_confusion_matrix(model_dir / "confusion_matrix.png")
        self.plot_roc_curve(X_test, y_test, model_dir / "roc_curve.png")
        self.plot_precision_recall_curve(X_test, y_test, model_dir / "precision_recall_curve.png")
        self.plot_feature_importance(feature_names, save_path=model_dir / "feature_importance.png")
        self.plot_threshold_analysis(X_test, y_test, model_dir / "threshold_analysis.png")
        
        # Save metrics to JSON
        import json
        metrics_to_save = {k: v for k, v in self.metrics.items() 
                          if k not in ['confusion_matrix', 'classification_report']}
        metrics_to_save['confusion_matrix'] = self.metrics['confusion_matrix'].tolist()
        
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        logger.info(f"Evaluation report completed. Files saved in {model_dir}")


if __name__ == "__main__":
    # Load data and model
    from data_loader import TelcoDataLoader
    from preprocessing import TelcoPreprocessor
    from model_trainer import ChurnModelTrainer
    
    # Load and preprocess data
    loader = TelcoDataLoader()
    df = loader.load_data()
    
    preprocessor = TelcoPreprocessor()
    X, y, feature_names = preprocessor.prepare_features(df)
    
    # Split data
    trainer = ChurnModelTrainer(random_state=42)
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # Load best model (try tuned first, then default)
    try:
        model = joblib.load("models/tuned/random_forest_tuned.pkl")
        model_name = "Random Forest (Tuned)"
        logger.info("Loaded tuned Random Forest model")
    except:
        model = joblib.load("models/random_forest.pkl")
        model_name = "Random Forest"
        logger.info("Loaded default Random Forest model")
    
    # Evaluate model
    evaluator = ModelEvaluator(model, model_name)
    evaluator.generate_evaluation_report(X_test, y_test, feature_names)