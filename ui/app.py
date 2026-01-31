import streamlit as st
import pandas as pd
import sys
import os
import time

# Add root directory to path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train import train_model

st.set_page_config(page_title="Cloud Cost MLOps Dashboard", layout="wide")

st.title("Cloud Cost Prediction System")

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        data_dir = "data/raw"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, "cloud_cost_data.csv")
        
        # Read file as dataframe to ensure it's valid CSV before saving
        df = pd.read_csv(uploaded_file)
        df.to_csv(file_path, index=False)
        return True, df
    except Exception as e:
        return False, e

# Sidebar
st.sidebar.header("Operations")
option = st.sidebar.radio("Select Action", ["Upload Data", "Train Model", "View Metrics"])

# Upload Data Section
if option == "Upload Data":
    st.header("Upload Historical Cloud Cost Data")
    st.info("Upload the CSV export from AWS Cost Explorer.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        if st.button("Save & Validate"):
            success, result = save_uploaded_file(uploaded_file)
            if success:
                st.success("File saved successfully to `data/raw/cloud_cost_data.csv`")
                st.dataframe(result.head())
            else:
                st.error(f"Error saving file: {result}")

# Train Model Section
elif option == "Train Model":
    st.header("Model Training Pipeline")
    
    if st.button("Start Training"):
        with st.status("Running Training Pipeline...", expanded=True) as status:
            try:
                st.write("Initializing data loader...")
                time.sleep(1)
                st.write("Starting training process...")
                
                # Quick fix: Transform data directly here
                import pandas as pd
                from sklearn.ensemble import RandomForestRegressor
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                import mlflow
                import mlflow.sklearn
                
                # Load and transform data
                df = pd.read_csv("data/raw/cloud_cost_data.csv")
                df = df[df['Service'] != 'Service total'].copy()
                df['date'] = pd.to_datetime(df['Service'])
                df['ec2_hours'] = pd.to_numeric(df.get('EC2-Other($)', 0), errors='coerce').fillna(0) * 24
                df['storage_gb'] = pd.to_numeric(df.get('S3($)', 0), errors='coerce').fillna(0) * 100
                df['data_transfer_gb'] = pd.to_numeric(df.get('CloudWatch($)', 0), errors='coerce').fillna(0) * 10
                df['rds_usage'] = pd.to_numeric(df.get('Relational Database Service($)', 0), errors='coerce').fillna(0)
                df['lambda_invocations'] = (pd.to_numeric(df.get('Lambda($)', 0), errors='coerce').fillna(0) * 1000000).astype(int)
                df['daily_cost'] = pd.to_numeric(df.get('Total costs($)', 0), errors='coerce').fillna(0)
                
                df = df[['date', 'ec2_hours', 'storage_gb', 'data_transfer_gb', 'rds_usage', 'lambda_invocations', 'daily_cost']]
                df = df.sort_values('date').reset_index(drop=True)
                
                # Simple features
                df['lag_1'] = df['daily_cost'].shift(1).fillna(0)
                df['rolling_3'] = df['daily_cost'].rolling(3).mean().shift(1).fillna(df['daily_cost'])
                df['rolling_7'] = df['daily_cost'].rolling(7).mean().shift(1).fillna(df['daily_cost'])
                df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
                
                # Train model
                X = df[['ec2_hours', 'storage_gb', 'data_transfer_gb', 'rds_usage', 'lambda_invocations', 'lag_1', 'rolling_3', 'rolling_7', 'is_weekend']]
                y = df['daily_cost']
                
                train_size = int(len(df) * 0.8)
                X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
                y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
                
                mlflow.set_tracking_uri("file:./mlruns")
                mlflow.set_experiment("cloud_cost_prediction")
                
                with mlflow.start_run():
                    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                    model.fit(X_train, y_train)
                    
                    predictions = model.predict(X_test)
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = mean_squared_error(y_test, predictions) ** 0.5
                    
                    mlflow.log_metric("mae", mae)
                    mlflow.log_metric("rmse", rmse)
                    mlflow.sklearn.log_model(model, "model", registered_model_name="cost_prediction_model")
                    
                    st.write(f"MAE: {mae:.2f}")
                    st.write(f"RMSE: {rmse:.2f}")
                    
                    # Send training completion notification
                    try:
                        from notifications.telegram import TelegramNotifier
                        notifier = TelegramNotifier()
                        notifier.send_training_alert({'mae': mae, 'rmse': rmse})
                    except Exception as e:
                        st.write(f"Notification failed: {e}")
                
                st.write("Model registered to MLflow.")
                status.update(label="Training Complete!", state="complete", expanded=False)
                st.success("Model trained successfully!")
                
            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                status.update(label="Training Failed", state="error")
                st.error(f"Training failed: {str(e)}")
                
                # Show detailed error in expandable section
                with st.expander("Detailed Error Log"):
                    st.code(error_details, language="python")
                    
                # Also show current working directory and file paths for debugging
                with st.expander("Debug Information"):
                    st.write(f"Current working directory: {os.getcwd()}")
                    st.write(f"Script location: {os.path.abspath(__file__)}")
                    data_path = "data/raw/cloud_cost_data.csv"
                    st.write(f"Looking for data at: {os.path.abspath(data_path)}")
                    st.write(f"Data file exists: {os.path.exists(data_path)}")
# Metrics Section
elif option == "View Metrics":
    st.header("Latest Model Metrics")
    st.info("Please visit the MLflow UI for detailed metrics and plots.")
    st.markdown("[Open MLflow UI >](http://localhost:5000)")
    
    st.markdown("### Quick Links")
    st.markdown("- [Prediction API Docs](http://localhost:8000/docs)")
    st.markdown("- [Grafana Dashboards](http://localhost:3000)")
    st.markdown("- [Prometheus](http://localhost:9090)")
