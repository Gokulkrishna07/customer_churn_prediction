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
option = st.sidebar.radio("Select Action", ["Upload Data", "Train Model", "Make Prediction", "Outputs", "View Metrics"])

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
                
                # Send CSV upload notification
                try:
                    from notifications.telegram import TelegramNotifier
                    notifier = TelegramNotifier()
                    
                    # Calculate summary stats
                    total_cost = result.get('Total costs($)', 0).sum() if 'Total costs($)' in result.columns else 0
                    num_records = len(result)
                    date_range = f"{result.iloc[0, 0]} to {result.iloc[-1, 0]}" if num_records > 0 else "Unknown"
                    
                    message = f"""
📊 <b>NEW CSV DATA UPLOADED</b>

📁 <b>File:</b> {uploaded_file.name}
📈 <b>Records:</b> {num_records}
📅 <b>Date Range:</b> {date_range}
💰 <b>Total Cost:</b> ${total_cost:.2f}

🕐 <b>Uploaded:</b> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ <b>Status:</b> Ready for training
                    """
                    
                    notifier.send_message(message)
                    st.success("📱 Notification sent to Telegram!")
                except Exception as e:
                    st.warning(f"Notification failed: {e}")
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
# Make Prediction Section
elif option == "Make Prediction":
    st.header("Cost Prediction")
    st.info("Enter your AWS usage metrics to get cost prediction and alerts.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        ec2_hours = st.number_input("EC2 Hours", min_value=0.0, value=100.0)
        storage_gb = st.number_input("Storage (GB)", min_value=0.0, value=500.0)
        data_transfer_gb = st.number_input("Data Transfer (GB)", min_value=0.0, value=50.0)
    
    with col2:
        rds_usage = st.number_input("RDS Usage", min_value=0.0, value=10.0)
        lambda_invocations = st.number_input("Lambda Invocations", min_value=0, value=1000000)
        budget = st.number_input("Daily Budget ($)", min_value=0.0, value=200.0)
    
    if st.button("Get Prediction"):
        try:
            import requests
            
            # Make API call
            payload = {
                "ec2_hours": ec2_hours,
                "storage_gb": storage_gb,
                "data_transfer_gb": data_transfer_gb,
                "rds_usage": rds_usage,
                "lambda_invocations": int(lambda_invocations),
                "budget": budget
            }
            
            response = requests.post("http://localhost:8000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Predicted Cost", f"${result['predicted_cost']:.2f}")
                
                with col2:
                    st.metric("Budget", f"${result['budget']:.2f}")
                
                with col3:
                    risk_color = "🔴" if result['risk_level'] == "High" else "🟡" if result['risk_level'] == "Medium" else "🟢"
                    st.metric("Risk Level", f"{risk_color} {result['risk_level']}")
                
                if result['overrun']:
                    st.error(f"⚠️ Budget Overrun Alert! Predicted cost exceeds budget by ${result['predicted_cost'] - result['budget']:.2f}")
                else:
                    st.success(f"✅ Within Budget! You have ${result['budget'] - result['predicted_cost']:.2f} remaining.")
                
                # Show notification status
                if result['overrun'] or result['risk_level'] == "High":
                    st.info("📱 Alert notification sent to Telegram!")
                
            else:
                st.error(f"Prediction failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Outputs Section
elif option == "Outputs":
    st.header("📊 Cost Prediction Outputs")
    
    # Quick prediction form
    with st.expander("🔮 Quick Cost Prediction", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ec2_hours = st.number_input("EC2 Hours", min_value=0.0, value=100.0, key="output_ec2")
            storage_gb = st.number_input("Storage (GB)", min_value=0.0, value=500.0, key="output_storage")
        
        with col2:
            data_transfer_gb = st.number_input("Data Transfer (GB)", min_value=0.0, value=50.0, key="output_transfer")
            rds_usage = st.number_input("RDS Usage", min_value=0.0, value=10.0, key="output_rds")
        
        with col3:
            lambda_invocations = st.number_input("Lambda Invocations", min_value=0, value=1000000, key="output_lambda")
            budget = st.number_input("Daily Budget ($)", min_value=0.0, value=200.0, key="output_budget")
        
        if st.button("🎯 Get Today's Cost Prediction", type="primary"):
            try:
                import requests
                from datetime import datetime
                
                # Make API call
                payload = {
                    "ec2_hours": ec2_hours,
                    "storage_gb": storage_gb,
                    "data_transfer_gb": data_transfer_gb,
                    "rds_usage": rds_usage,
                    "lambda_invocations": int(lambda_invocations),
                    "budget": budget
                }
                
                response = requests.post("http://localhost:8000/predict", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_cost = result['predicted_cost']
                    
                    # Main prediction display
                    st.markdown("---")
                    st.markdown("### 💰 Today's Cost Prediction")
                    
                    # Big cost display
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        if result['overrun']:
                            st.error(f"🚨 **BUDGET ALERT!**")
                            st.markdown(f"### Your predicted cost today: **${predicted_cost:.2f}**")
                            st.markdown(f"💸 **Over budget by:** ${predicted_cost - budget:.2f}")
                        else:
                            st.success(f"✅ **WITHIN BUDGET**")
                            st.markdown(f"### Your predicted cost today: **${predicted_cost:.2f}**")
                            st.markdown(f"💚 **Remaining budget:** ${budget - predicted_cost:.2f}")
                    
                    with col2:
                        # Risk indicator
                        risk_color = "🔴" if result['risk_level'] == "High" else "🟡" if result['risk_level'] == "Medium" else "🟢"
                        st.markdown(f"### {risk_color}")
                        st.markdown(f"**{result['risk_level']} Risk**")
                    
                    with col3:
                        # Budget progress bar
                        progress = min(predicted_cost / budget, 1.0)
                        st.markdown("### Budget Usage")
                        st.progress(progress)
                        st.markdown(f"**{progress*100:.1f}%** of budget")
                    
                    # Detailed breakdown
                    st.markdown("---")
                    st.markdown("### 📈 Cost Breakdown Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**📊 Usage Summary:**")
                        st.write(f"• EC2 Compute: {ec2_hours} hours")
                        st.write(f"• Storage: {storage_gb:,.0f} GB")
                        st.write(f"• Data Transfer: {data_transfer_gb} GB")
                        st.write(f"• RDS Usage: {rds_usage}")
                        st.write(f"• Lambda Calls: {lambda_invocations:,}")
                    
                    with col2:
                        st.markdown("**🎯 Prediction Details:**")
                        st.write(f"• Model Version: {result['model_version']}")
                        st.write(f"• Prediction Time: {datetime.now().strftime('%H:%M:%S')}")
                        st.write(f"• Risk Assessment: {result['risk_level']}")
                        if result['overrun']:
                            st.write("• 🚨 Alert: Budget exceeded!")
                        else:
                            st.write("• ✅ Status: Within budget")
                    
                    # Recommendations
                    st.markdown("---")
                    st.markdown("### 💡 Recommendations")
                    
                    if result['overrun']:
                        st.warning("**Cost Optimization Needed:**")
                        st.write("• Consider reducing EC2 instance hours")
                        st.write("• Optimize storage usage and cleanup unused data")
                        st.write("• Review data transfer patterns")
                        st.write("• Scale down RDS instances if possible")
                    elif result['risk_level'] == "High":
                        st.info("**Monitor Closely:**")
                        st.write("• You're approaching your budget limit")
                        st.write("• Consider setting up cost alerts")
                        st.write("• Review usage patterns for optimization")
                    else:
                        st.success("**Good Cost Management:**")
                        st.write("• Your usage is well within budget")
                        st.write("• Continue monitoring daily costs")
                        st.write("• Consider increasing budget for growth")
                    
                    # Save prediction to session state for history
                    if 'prediction_history' not in st.session_state:
                        st.session_state.prediction_history = []
                    
                    st.session_state.prediction_history.append({
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'predicted_cost': predicted_cost,
                        'budget': budget,
                        'risk_level': result['risk_level'],
                        'overrun': result['overrun']
                    })
                    
                else:
                    st.error(f"❌ Prediction failed: {response.text}")
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
    
    # Prediction History
    if 'prediction_history' in st.session_state and st.session_state.prediction_history:
        st.markdown("---")
        with st.expander("📋 Recent Predictions History"):
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df, use_container_width=True)
            
            if st.button("🗑️ Clear History"):
                st.session_state.prediction_history = []
                st.rerun()

# Metrics Section
elif option == "View Metrics":
    st.header("Latest Model Metrics")
    st.info("Please visit the MLflow UI for detailed metrics and plots.")
    st.markdown("[Open MLflow UI >](http://localhost:5000)")
    
    st.markdown("### Quick Links")
    st.markdown("- [Prediction API Docs](http://localhost:8000/docs)")
    st.markdown("- [Grafana Dashboards](http://localhost:3000)")
    st.markdown("- [Prometheus](http://localhost:9090)")
