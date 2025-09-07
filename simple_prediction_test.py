#!/usr/bin/env python3
"""
Simple prediction test script demonstrating both single model and ensemble approaches.

This script shows:
1. How to train and use a single Random Forest model
2. How to use existing ensemble models (LightGBM + XGBoost)
3. Compare individual models (LightGBM or XGBoost) vs ensemble
4. Compare their predictions
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.prediction.features import create_features
from src.prediction.predictor import CryptoPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def example_1_single_model():
    """Example 1: Train and use a single Random Forest model."""
    logger.info("=" * 60)
    logger.info("EXAMPLE 1: SINGLE MODEL PREDICTION")
    logger.info("=" * 60)

    # Step 1: Create and train a single model
    logger.info("Step 1: Training a single Random Forest model...")

    # Create directory for single model
    model_dir = Path("models_single_example")
    model_dir.mkdir(exist_ok=True)

    # Load and prepare training data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")
    df_features = create_features(df)

    # Define features
    feature_names = [
        "minute_in_hour",
        "ret_5m",
        "ret_15m",
        "ret_30m",
        "ret_1h",
        "ret_2h",
        "ret_3h",
        "price_deviation_from_open",
        "relative_price_position",
        "vol_share_in_hour",
        "em5_ratio",
        "em10_ratio",
        "em20_ratio",
        "em30_ratio",
        "ema5_ema10_ratio",
        "ema5_ema20_ratio",
        "ema5_ema30_ratio",
        "ema10_ema20_ratio",
        "ema10_ema30_ratio",
        "ema20_ema30_ratio",
        "rsi_5",
        "rsi_10",
        "rsi_20",
        "rsi_30",
    ]

    # Create target variable (1 if next price goes up, 0 otherwise)
    df_features = df_features.with_columns(
        [(pl.col("close").shift(-1) > pl.col("close")).cast(pl.Int32).alias("target")]
    ).drop_nulls()

    # Prepare features and target
    X = df_features.select(feature_names).to_numpy()
    y = df_features["target"].to_numpy()
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train scaler and model
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred_proba_matrix = rf_model.predict_proba(X_test_scaled)
    y_pred_proba = [prob[1] for prob in y_pred_proba_matrix]
    y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred_proba]
    accuracy = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    logger.info(f"Model Performance: Accuracy={accuracy:.3f}, ROC-AUC={roc_auc:.3f}")

    # Save model artifacts
    joblib.dump(rf_model, model_dir / "randomforest_model.joblib")
    joblib.dump(scaler, model_dir / "scaler.joblib")
    with open(model_dir / "features.json", "w") as f:
        json.dump(feature_names, f)

    logger.info(f"Model saved to: {model_dir}")

    # Step 2: Use the trained model for prediction
    logger.info("\nStep 2: Making predictions with single model...")

    predictor = CryptoPredictor(models_dir=str(model_dir))
    prediction_prob = predictor.predict(df)

    logger.info("Single Model Results:")
    logger.info(f"  Probability of price going UP: {prediction_prob:.4f}")
    logger.info(f"  Prediction: {'BUY' if prediction_prob > 0.5 else 'SELL'}")
    logger.info(f"  Confidence: {max(prediction_prob, 1 - prediction_prob):.4f}")

    return prediction_prob


def example_2_ensemble_models():
    """Example 2: Use existing ensemble models (LightGBM + XGBoost)."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 2: ENSEMBLE PREDICTION")
    logger.info("=" * 60)

    # Check if ensemble models exist
    models_dir = Path("models")
    model_files = list(models_dir.glob("*_model.joblib"))

    if not model_files:
        logger.error("No ensemble models found! Please train models first.")
        return None

    logger.info(f"Found ensemble models: {[f.stem.replace('_model', '') for f in model_files]}")

    # Load data and make prediction
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

    # Use ensemble predictor
    predictor = CryptoPredictor(models_dir=str(models_dir))
    prediction_prob = predictor.predict(df)

    logger.info("Ensemble Model Results:")
    logger.info(f"  Models used: {list(predictor.models.keys())}")
    logger.info(f"  Average probability of price going UP: {prediction_prob:.4f}")
    logger.info(f"  Prediction: {'BUY' if prediction_prob > 0.5 else 'SELL'}")
    logger.info(f"  Confidence: {max(prediction_prob, 1 - prediction_prob):.4f}")

    return prediction_prob


def example_3_individual_vs_ensemble():
    """Example 3: Compare individual models (LightGBM, XGBoost) vs ensemble."""
    logger.info("\n" + "=" * 60)
    logger.info("EXAMPLE 3: INDIVIDUAL MODELS vs ENSEMBLE")
    logger.info("=" * 60)

    # Check if ensemble models exist
    models_dir = Path("models")
    model_files = list(models_dir.glob("*_model.joblib"))

    if not model_files:
        logger.error("No ensemble models found! Please train models first.")
        return None

    logger.info(f"Found ensemble models: {[f.stem.replace('_model', '') for f in model_files]}")

    # Load data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

    # Use ensemble predictor
    predictor = CryptoPredictor(models_dir=str(models_dir))

    # Get individual predictions
    individual_predictions = predictor.get_individual_predictions(df)
    ensemble_prob = predictor.predict(df)

    logger.info("\nModel Predictions:")
    logger.info("-" * 50)

    results = {}
    for model_name, prob in individual_predictions.items():
        direction = "BUY" if prob > 0.5 else "SELL"
        confidence = max(prob, 1 - prob)
        confidence_text = f"Confidence: {confidence:.4f}"
        logger.info(f"{model_name.upper():>10}: {prob:.4f} ({direction}) - {confidence_text}")
        results[model_name] = {"prob": prob, "direction": direction, "confidence": confidence}

    # Ensemble results
    ensemble_direction = "BUY" if ensemble_prob > 0.5 else "SELL"
    ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob)
    logger.info(
        f"{'ENSEMBLE':>10}: {ensemble_prob:.4f} ({ensemble_direction}) - Confidence: {ensemble_confidence:.4f}"
    )

    # Detailed comparisons
    logger.info("\nDetailed Comparisons:")
    logger.info("-" * 50)

    for model_name, model_results in results.items():
        prob_diff = abs(model_results["prob"] - ensemble_prob)
        direction_agreement = "Yes" if model_results["direction"] == ensemble_direction else "No"

        logger.info(f"\n{model_name.upper()} vs ENSEMBLE:")
        individual_prob = model_results["prob"]
        individual_direction = model_results["direction"]
        logger.info(f"  Individual:  {individual_prob:.4f} ({individual_direction})")
        logger.info(f"  Ensemble:    {ensemble_prob:.4f} ({ensemble_direction})")
        logger.info(f"  Difference:  {prob_diff:.4f}")
        logger.info(f"  Agreement:   {direction_agreement}")

        if prob_diff < 0.05:
            logger.info(f"  🟢 {model_name.upper()} is very close to ensemble")
        elif prob_diff < 0.15:
            logger.info(f"  🟡 {model_name.upper()} shows moderate agreement")
        else:
            logger.info(f"  🔴 {model_name.upper()} shows significant disagreement")

    return results


def compare_results(single_prob, ensemble_prob):
    """Compare results from both approaches."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON OF RESULTS")
    logger.info("=" * 60)

    if single_prob is None or ensemble_prob is None:
        logger.warning("Cannot compare - one or both predictions failed")
        return

    single_direction = "BUY" if single_prob > 0.5 else "SELL"
    ensemble_direction = "BUY" if ensemble_prob > 0.5 else "SELL"

    single_confidence = max(single_prob, 1 - single_prob)
    ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob)

    logger.info(
        f"Single Model:   {single_prob:.4f} ({single_direction}) - Confidence: {single_confidence:.4f}"
    )
    logger.info(
        f"Ensemble Model: {ensemble_prob:.4f} ({ensemble_direction}) - Confidence: {ensemble_confidence:.4f}"
    )
    logger.info(f"Probability Difference: {abs(single_prob - ensemble_prob):.4f}")
    logger.info(f"Direction Agreement: {'YES' if single_direction == ensemble_direction else 'NO'}")

    # Interpretation
    if single_confidence > 0.8 and ensemble_confidence > 0.8:
        if single_direction == ensemble_direction:
            logger.info("🟢 Strong agreement with high confidence - Consider this signal!")
        else:
            logger.info("🟡 High confidence but different directions - Exercise caution!")
    elif single_confidence > 0.7 or ensemble_confidence > 0.7:
        logger.info("🟡 Moderate confidence - Consider additional factors")
    else:
        logger.info("🔴 Low confidence predictions - Avoid trading on this signal")


def main():
    """Run all examples and compare results."""
    logger.info("CRYPTOCURRENCY PREDICTION EXAMPLES")
    logger.info(
        "This script demonstrates single model, ensemble, and individual model comparisons\n"
    )

    try:
        # Run Example 1: Single Model
        single_prob = example_1_single_model()

        # Run Example 2: Ensemble Models
        ensemble_prob = example_2_ensemble_models()

        # Run Example 3: Individual vs Ensemble
        individual_results = example_3_individual_vs_ensemble()

        # Compare Results
        compare_results(single_prob, ensemble_prob)

        logger.info("\n" + "=" * 60)
        logger.info("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        # Final summary
        if single_prob and ensemble_prob and individual_results:
            logger.info("\nFINAL SUMMARY:")
            single_direction = "BUY" if single_prob > 0.5 else "SELL"
            logger.info(f"Single Model (RF): {single_prob:.4f} ({single_direction})")

            ensemble_direction = "BUY" if ensemble_prob > 0.5 else "SELL"
            logger.info(f"Ensemble Model:    {ensemble_prob:.4f} ({ensemble_direction})")

            for model_name, results in individual_results.items():
                model_prob = results["prob"]
                model_direction = results["direction"]
                logger.info(f"{model_name.upper():>14}: {model_prob:.4f} ({model_direction})")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error("Make sure you have the required data files and dependencies installed.")


if __name__ == "__main__":
    main()
