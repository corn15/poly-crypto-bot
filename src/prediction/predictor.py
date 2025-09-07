import json
import logging
import warnings
from pathlib import Path
from typing import Dict, Union

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
        """Prepare features for prediction."""
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

    def predict_single(self, df: pl.DataFrame) -> Dict[str, Dict[str, float]]:
        """Make prediction for a single sample (latest data point)."""
        X = self.prepare_features(df)

        if len(X) == 0:
            raise ValueError("No valid features could be created from the input data")

        # Use the last row (most recent data)
        X_latest = X[-1:, :]

        predictions = {}
        for model_name, model in self.models.items():
            # Get probability predictions
            prob = model.predict_proba(X_latest)[0, 1]  # Probability of class 1 (price up)
            pred = int(prob > 0.5)  # Binary prediction

            predictions[model_name] = {"probability": float(prob), "prediction": pred}

        return predictions

    def predict_batch(self, df: pl.DataFrame) -> Dict[str, np.ndarray]:
        """Make predictions for all valid data points in the dataframe."""
        X = self.prepare_features(df)

        if len(X) == 0:
            raise ValueError("No valid features could be created from the input data")

        predictions = {}
        for model_name, model in self.models.items():
            # Get probability predictions for all samples
            probs = model.predict_proba(X)[:, 1]  # Probabilities of class 1
            preds = (probs > 0.5).astype(int)  # Binary predictions

            predictions[model_name] = {"probabilities": probs, "predictions": preds}

        return predictions

    def predict_from_file(
        self, file_path: Union[str, Path], mode: str = "single"
    ) -> Dict[str, Union[Dict, np.ndarray]]:
        """Load data from file and make predictions."""
        file_path = Path(file_path)

        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")

        logger.info(f"Loading data from {file_path}")

        # Load data (assuming parquet format)
        if file_path.suffix == ".parquet":
            df = pl.read_parquet(file_path)
        elif file_path.suffix == ".csv":
            df = pl.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

        logger.info(f"Loaded {len(df)} rows from {file_path}")

        # Make predictions
        if mode == "single":
            return self.predict_single(df)  # type: ignore
        elif mode == "batch":
            return self.predict_batch(df)  # type: ignore
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'single' or 'batch'")

    def get_ensemble_prediction(self, predictions: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Create ensemble prediction by averaging model predictions."""
        if not predictions:
            return {}

        # Average probabilities across models
        avg_prob = np.mean([pred["probability"] for pred in predictions.values()])
        avg_pred = int(avg_prob > 0.5)

        return {"probability": float(avg_prob), "prediction": avg_pred}

    def print_predictions(self, predictions: Dict, ensemble: bool = True) -> None:
        """Print formatted predictions."""
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION RESULTS")
        logger.info("=" * 60)

        for model_name, pred in predictions.items():
            if isinstance(pred, dict) and "probability" in pred:
                prob = pred["probability"]
                prediction = pred["prediction"]
                direction = "UP" if prediction == 1 else "DOWN"
                confidence = max(prob, 1 - prob)

                logger.info(f"\n{model_name.upper()} Model:")
                logger.info(f"  Probability (UP): {prob:.4f}")
                logger.info(f"  Prediction: {direction}")
                logger.info(f"  Confidence: {confidence:.4f}")

        if ensemble and len(predictions) > 1:
            ensemble_pred = self.get_ensemble_prediction(predictions)
            prob = ensemble_pred["probability"]
            prediction = ensemble_pred["prediction"]
            direction = "UP" if prediction == 1 else "DOWN"
            confidence = max(prob, 1 - prob)

            logger.info("\nENSEMBLE (Average):")
            logger.info(f"  Probability (UP): {prob:.4f}")
            logger.info(f"  Prediction: {direction}")
            logger.info(f"  Confidence: {confidence:.4f}")

        logger.info("\n" + "=" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Predict crypto price direction")
    parser.add_argument(
        "--data", "-d", type=str, required=True, help="Path to data file (parquet or csv)"
    )
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["single", "batch"],
        default="single",
        help="Prediction mode: single (latest) or batch (all)",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing trained models",
    )

    args = parser.parse_args()

    try:
        # Initialize predictor
        predictor = CryptoPredictor(models_dir=args.models_dir)

        # Make predictions
        predictions = predictor.predict_from_file(args.data, mode=args.mode)

        # Print results
        if args.mode == "single":
            predictor.print_predictions(predictions)
        else:
            # For batch mode, print summary statistics
            logger.info("\nBatch Prediction Summary:")
            for model_name, pred_data in predictions.items():
                probs = pred_data["probabilities"]
                preds = pred_data["predictions"]

                logger.info(f"\n{model_name.upper()} Model:")
                logger.info(f"  Total predictions: {len(probs)}")
                logger.info(f"  Average probability: {np.mean(probs):.4f}")
                logger.info(f"  Predictions UP: {np.sum(preds)} ({np.mean(preds):.1%})")
                logger.info(f"  Predictions DOWN: {np.sum(1 - preds)} ({np.mean(1 - preds):.1%})")

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
