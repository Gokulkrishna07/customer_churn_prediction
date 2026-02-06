"""
Simple prediction script for customer churn.
Usage: python predict.py (run from project root)
"""

import sys
import os

# Add training directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))

from training.final_model import ProductionChurnModel

# Load the production model (correct path in training folder)
print("Loading model...")
model = ProductionChurnModel.load("training/models/production/churn_model_production.pkl")
print("✓ Model loaded successfully!\n")

# Example customer data
customer = {
    'gender': 'Female',
    'SeniorCitizen': 1,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 5,
    'PhoneService': 'Yes',
    'MultipleLines': 'Yes',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 10.15,
    'TotalCharges': 39.60
}

# Make prediction
print("Making prediction...")
result = model.predict_single(customer)

# Display results
print("\n" + "=" * 70)
print("CUSTOMER CHURN PREDICTION")
print("=" * 70)
print(f"\nCustomer Profile:")
print(f"   • Tenure: {customer['tenure']} months")
print(f"   • Contract: {customer['Contract']}")
print(f"   • Monthly Charges: ${customer['MonthlyCharges']}")
print(f"   • Total Charges: ${customer['TotalCharges']}")
print(f"   • Internet Service: {customer['InternetService']}")
print(f"   • Payment Method: {customer['PaymentMethod']}")

print(f"\nPrediction Results:")
print(f"   • Churn Risk: {'HIGH RISK' if result['churn_prediction'] == 1 else 'LOW RISK'}")
print(f"   • Churn Probability: {result['churn_probability']:.1%}")
print(f"   • Confidence: {result['confidence']:.1%}")

if result['churn_prediction'] == 1:
    print(f"\nRECOMMENDATION:")
    print(f"   This customer is at HIGH RISK of churning!")
    print(f"   Suggested actions:")
    print(f"   1. Offer retention incentives")
    print(f"   2. Contact customer proactively")
    print(f"   3. Review service satisfaction")
    print(f"   4. Consider contract upgrade offers")
else:
    print(f"\nRECOMMENDATION:")
    print(f"   This customer has LOW churn risk.")
    print(f"   Continue regular engagement.")

print("\n" + "=" * 70)

# Show top features
print(f"\nTop 5 Features Influencing Churn:")
top_features = model.get_feature_importance(5)
for i, (feature, importance) in enumerate(top_features.items(), 1):
    print(f"   {i}. {feature}: {importance:.2%}")
print("\n" + "=" * 70)