import requests
import os
from datetime import datetime

class TelegramNotifier:
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
    
    def send_message(self, message):
        """Send message to Telegram"""
        if not self.bot_token or not self.chat_id:
            print("Telegram credentials not configured")
            return False
            
        url = f"{self.base_url}/sendMessage"
        data = {
            'chat_id': self.chat_id,
            'text': message,
            'parse_mode': 'HTML'
        }
        
        try:
            response = requests.post(url, data=data)
            return response.status_code == 200
        except Exception as e:
            print(f"Failed to send Telegram message: {e}")
            return False
    
    def send_cost_alert(self, prediction_data):
        """Send cost prediction alert"""
        predicted_cost = prediction_data['predicted_cost']
        budget = prediction_data['budget']
        risk_level = prediction_data['risk_level']
        overrun = prediction_data['overrun']
        
        # Determine alert type
        if overrun:
            emoji = "🚨"
            status = "BUDGET OVERRUN ALERT"
        elif risk_level == "High":
            emoji = "⚠️"
            status = "HIGH COST WARNING"
        elif risk_level == "Medium":
            emoji = "📊"
            status = "COST UPDATE"
        else:
            emoji = "✅"
            status = "COST NORMAL"
        
        message = f"""
{emoji} <b>{status}</b>

💰 <b>Predicted Cost:</b> ${predicted_cost:.2f}
🎯 <b>Budget:</b> ${budget:.2f}
📈 <b>Risk Level:</b> {risk_level}
{'💸 <b>Over Budget:</b> Yes' if overrun else '✅ <b>Within Budget:</b> Yes'}

🕐 <b>Time:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
🤖 <b>Model:</b> {prediction_data.get('model_version', 'v1')}
        """
        
        return self.send_message(message)
    
    def send_training_alert(self, metrics):
        """Send model training completion alert"""
        message = f"""
🎯 <b>MODEL TRAINING COMPLETED</b>

📊 <b>Performance Metrics:</b>
• MAE: ${metrics.get('mae', 0):.2f}
• RMSE: ${metrics.get('rmse', 0):.2f}

🕐 <b>Completed:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ <b>Status:</b> Model ready for predictions
        """
        
        return self.send_message(message)