import json
import logging
import warnings
from pathlib import Path

import joblib
import numpy as np
import polars as pl

from src.prediction.features import create_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


class CryptoPredictor:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scaler = None
        self.feature_names = []
        self.load_artifacts()

    def load_artifacts(self) -> None:
        logger.info("Loading trained models and artifacts...")

        # Load models
        for model_file in self.models_dir.glob("*_model.joblib"):
            model_name = model_file.stem.replace("_model", "")
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded {model_name} model")

        if not self.models:
            raise ValueError(f"No models found in {self.models_dir}")

        # Load scaler
        scaler_file = self.models_dir / "scaler.joblib"
        if scaler_file.exists():
            self.scaler = joblib.load(scaler_file)
            logger.info("Loaded feature scaler")
        else:
            raise ValueError(f"Scaler not found: {scaler_file}")

        # Load feature names
        features_file = self.models_dir / "features.json"
        if features_file.exists():
            with open(features_file, "r") as f:
                self.feature_names = json.load(f)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
        else:
            raise ValueError(f"Feature names not found: {features_file}")

    def prepare_features(self, df: pl.DataFrame) -> np.ndarray:
        # Create all features
        df = create_features(df)

        # Select only the features used in training
        df_features = df.select(self.feature_names).drop_nulls()

        # Convert to numpy array
        X = df_features.to_numpy()

        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        # Scale features using the fitted scaler
        X_scaled = X
        if self.scaler:
            X_scaled = self.scaler.transform(X)

        return X_scaled

    def predict(self, df: pl.DataFrame) -> float:
        """Return ensemble prediction by averaging model predictions."""
        X = self.prepare_features(df)

        if len(X) == 0:
            raise ValueError("No valid features could be created from the input data")

        # Use the last row (most recent data)
        X_latest = X[-1:, :]

        predictions = []
        for model_name, model in self.models.items():
            # Get probability predictions
            prob = model.predict_proba(X_latest)[0, 1]  # Probability of class 1 (price up)
            predictions.append(float(prob))

        avg_prob = float(np.mean(predictions))
        return avg_prob

    def predict_single_model(self, df: pl.DataFrame, model_name: str) -> float:
        """Return prediction from a specific model."""
        if model_name not in self.models:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
            )

        X = self.prepare_features(df)

        if len(X) == 0:
            raise ValueError("No valid features could be created from the input data")

        # Use the last row (most recent data)
        X_latest = X[-1:, :]

        model = self.models[model_name]
        prob = model.predict_proba(X_latest)[0, 1]  # Probability of class 1 (price up)

        return float(prob)

    def get_individual_predictions(self, df: pl.DataFrame) -> dict:
        """Return predictions from all individual models."""
        X = self.prepare_features(df)

        if len(X) == 0:
            raise ValueError("No valid features could be created from the input data")

        # Use the last row (most recent data)
        X_latest = X[-1:, :]

        predictions = {}
        for model_name, model in self.models.items():
            prob = model.predict_proba(X_latest)[0, 1]  # Probability of class 1 (price up)
            predictions[model_name] = float(prob)

        return predictions
