import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.prediction.features import FEATURES, create_features

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


class CryptoPredictor:
    def __init__(self, data_dir: str = "data", model_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}

    def load_data(self, symbols: Optional[List[str]] = None) -> pl.DataFrame:
        """Load and combine data from multiple symbols."""
        if symbols is None:
            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

        logger.info(f"Loading data for symbols: {symbols}")

        dfs = []
        for symbol in symbols:
            file_path = self.data_dir / f"{symbol}_5m_20240901.parquet"
            if file_path.exists():
                df = pl.read_parquet(file_path)
                df = df.with_columns([pl.lit(symbol).alias("symbol")])
                dfs.append(df)
                logger.info(f"Loaded {len(df)} rows for {symbol}")
            else:
                logger.warning(f"Data file not found: {file_path}")

        if not dfs:
            raise ValueError("No data files found!")

        # Combine all dataframes
        combined_df = pl.concat(dfs)
        logger.info(f"Combined dataset shape: {combined_df.shape}")

        return combined_df

    def create_target(self, df: pl.DataFrame) -> pl.DataFrame:
        """Create target variable: 1 if hourly close > hourly open, 0 otherwise."""
        logger.info("Creating target variable...")

        # Group by symbol and hour to get hourly open and close
        df = df.with_columns([pl.col("open_datetime").dt.truncate("1h").alias("hour_start")])

        # Get first open and last close for each hour
        hourly_data = df.group_by(["symbol", "hour_start"]).agg(
            [
                pl.col("open").first().alias("hour_open"),
                pl.col("close").last().alias("hour_close"),
            ]
        )

        # Create target: 1 if close > open, 0 otherwise
        hourly_data = hourly_data.with_columns(
            [(pl.col("hour_close") > pl.col("hour_open")).cast(pl.Int32).alias("target")]
        )

        # Join back with original data
        df = df.join(
            hourly_data.select(["symbol", "hour_start", "target"]),
            on=["symbol", "hour_start"],
            how="left",
        )

        logger.info(f"Target distribution: {df['target'].value_counts()}")
        return df

    def prepare_data(self, df: pl.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("Preparing data for training...")

        # Create features
        df = create_features(df)

        # Create target
        df = self.create_target(df)

        # Filter out rows with missing values in features or target
        feature_cols = FEATURES + ["target"]
        df_clean = df.select(feature_cols).drop_nulls()

        logger.info(f"Clean dataset shape: {df_clean.shape}")

        # Convert to numpy arrays
        X = df_clean.select(FEATURES).to_numpy()
        y = df_clean.select("target").to_numpy().flatten()

        # Handle any remaining NaN or inf values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

        logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"Target balance: {np.mean(y):.3f}")

        return X, y

    def train_lightgbm(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> lgb.LGBMClassifier:
        """Train LightGBM model."""
        logger.info("Training LightGBM model...")

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            boosting_type="gbdt",
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=0,
            random_state=42,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
        )

        return model

    def train_xgboost(
        self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray
    ) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=0.05,
            max_depth=6,
            n_estimators=1000,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            eval_metric="logloss",
            early_stopping_rounds=100,
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

        return model

    def evaluate_model(
        self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info(f"Evaluating {model_name}...")

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "log_loss": log_loss(y_test, y_pred_proba),
        }

        logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def get_feature_importance(
        self, model, feature_names: List[str], model_name: str, top_n: int = 20
    ) -> Dict[str, float]:
        """Get and log feature importance."""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "feature_importance"):
            importance = model.feature_importance()
        else:
            logger.warning(f"Cannot get feature importance for {model_name}")
            return {}

        feature_importance = dict(zip(feature_names, importance))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        logger.info(f"\nTop {top_n} features for {model_name}:")
        for i, (feature, importance_val) in enumerate(sorted_features[:top_n], 1):
            logger.info(f"  {i:2d}. {feature}: {importance_val:.4f}")

        return feature_importance

    def save_models(self) -> None:
        """Save trained models and metadata."""
        models_path = Path(self.model_dir)
        models_path.mkdir(exist_ok=True)

        # Save models
        for model_name, model in self.models.items():
            model_file = models_path / f"{model_name}_model.joblib"
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} model to {model_file}")

        # Save scaler
        scaler_file = models_path / "scaler.joblib"
        joblib.dump(self.scaler, scaler_file)

        # Save feature names
        features_file = models_path / "features.json"
        with open(features_file, "w") as f:
            json.dump(FEATURES, f)

        # Save results (convert numpy types to native Python types)
        results_serializable = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                results_serializable[key] = {
                    k: float(v) if hasattr(v, "item") else v for k, v in value.items()
                }
            else:
                results_serializable[key] = value

        results_file = models_path / "results.json"
        with open(results_file, "w") as f:
            json.dump(results_serializable, f, indent=2)

        logger.info(f"Saved all artifacts to {models_path}")

    def train(self, test_size: float = 0.2, val_size: float = 0.1):
        """Main training pipeline."""
        logger.info("Starting training pipeline...")

        # Load data
        df = self.load_data()

        # Prepare features and target
        X, y = self.prepare_data(df)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size / (1 - test_size), random_state=42, stratify=y_temp
        )

        logger.info(
            f"Train size: {X_train.shape[0]}, Val size: {X_val.shape[0]}, Test size: {X_test.shape[0]}"  # type: ignore # noqa: E501
        )

        # Train models
        lgb_model = self.train_lightgbm(X_train, y_train, X_val, y_val)  # type: ignore
        xgb_model = self.train_xgboost(X_train, y_train, X_val, y_val)  # type: ignore

        self.models["lightgbm"] = lgb_model
        self.models["xgboost"] = xgb_model

        # Evaluate models
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test, model_name)  # type: ignore
            self.results[model_name] = metrics

            # Feature importance
            feature_importance = self.get_feature_importance(model, FEATURES, model_name)
            self.results[f"{model_name}_feature_importance"] = feature_importance

        # Save everything
        self.save_models()

        logger.info("Training completed successfully!")

        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 50)

        for model_name in self.models.keys():
            metrics = self.results[model_name]
            logger.info(f"\n{model_name.upper()} Results:")
            logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
            logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--data_dir", type=str, default="data")
    args.add_argument("--model_dir", type=str, default="models")
    args = args.parse_args()

    predictor = CryptoPredictor(args.data_dir, args.model_dir)
    predictor.train()


if __name__ == "__main__":
    main()
