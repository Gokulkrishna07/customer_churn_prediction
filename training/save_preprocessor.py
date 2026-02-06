"""
Quick script to save the fitted preprocessor.
"""

from data_loader import TelcoDataLoader
from preprocessing import TelcoPreprocessor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data
loader = TelcoDataLoader()
df = loader.load_data()

# Fit and save preprocessor
preprocessor = TelcoPreprocessor()
X, y, feature_names = preprocessor.prepare_features(df)

# Save it
preprocessor.save("models/preprocessor.pkl")

logger.info("✓ Preprocessor saved successfully!")