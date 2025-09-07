#!/usr/bin/env python3
"""
Example script demonstrating how to use the cryptocurrency price prediction models.

This script shows two main usage patterns:
1. Single model prediction using a newly trained model
2. Ensemble prediction using existing LightGBM and XGBoost models
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


def create_single_model_example():
    """Create a simple single model for demonstration purposes."""
    logger.info("=== Creating Single Model Example ===")

    # Create directory for single model
    single_model_dir = Path("models_single_rf")
    single_model_dir.mkdir(exist_ok=True)

    # Load data for training
    logger.info("Loading training data...")
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

    # Create features and target
    df_features = create_features(df)

    # Select features (using same features as the ensemble models)
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

    # Create target (1 if next period price goes up, 0 otherwise)
    df_features = df_features.with_columns(
        [(pl.col("close").shift(-1) > pl.col("close")).cast(pl.Int32).alias("target")]
    ).drop_nulls()

    # Prepare features and target
    X = df_features.select(feature_names).to_numpy()
    y = df_features["target"].to_numpy()

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and fit scaler
    logger.info("Training scaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest model
    logger.info("Training Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_scaled, y_train)

    # Evaluate model
    y_pred = rf_model.predict(X_test_scaled)
    y_pred_proba_matrix = rf_model.predict_proba(X_test_scaled)
    y_pred_proba = [prob[1] for prob in y_pred_proba_matrix]  # Get probability of positive class

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    logger.info("Random Forest Model Performance:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  ROC AUC: {roc_auc:.4f}")

    # Save model artifacts
    logger.info("Saving model artifacts...")

    # Save model
    joblib.dump(rf_model, single_model_dir / "randomforest_model.joblib")

    # Save scaler
    joblib.dump(scaler, single_model_dir / "scaler.joblib")

    # Save feature names
    with open(single_model_dir / "features.json", "w") as f:
        json.dump(feature_names, f)

    logger.info(f"Single model saved to {single_model_dir}")
    return single_model_dir


def example_single_model_prediction():
    """Example 1: Single model prediction using newly created Random Forest model."""
    logger.info("\n=== Example 1: Single Model Prediction ===")

    # Create single model if it doesn't exist
    single_model_dir = Path("models_single_rf")
    if not single_model_dir.exists() or not list(single_model_dir.glob("*_model.joblib")):
        single_model_dir = create_single_model_example()

    # Initialize predictor with single model
    predictor = CryptoPredictor(models_dir=str(single_model_dir))

    # Load test data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")
    logger.info(f"Loaded {len(df)} rows of BTC data")

    # Make prediction using the latest data
    prediction_prob = predictor.predict(df)
    prediction_class = 1 if prediction_prob > 0.5 else 0
    confidence = max(prediction_prob, 1 - prediction_prob)
    direction = "UP" if prediction_class == 1 else "DOWN"

    logger.info("\nSingle Model (Random Forest) Results:")
    logger.info(f"  Probability of price going UP: {prediction_prob:.4f}")
    logger.info(f"  Predicted direction: {direction}")
    logger.info(f"  Confidence: {confidence:.4f}")

    return prediction_prob, prediction_class


def example_ensemble_prediction():
    """Example 2: Ensemble prediction using existing LightGBM and XGBoost models."""
    logger.info("\n=== Example 2: Ensemble Prediction ===")

    # Check if ensemble models exist
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("Models directory not found. Please train models first.")
        return None, None

    model_files = list(models_dir.glob("*_model.joblib"))
    if not model_files:
        logger.error("No trained models found. Please train models first.")
        return None, None

    logger.info(f"Found models: {[f.stem.replace('_model', '') for f in model_files]}")

    # Initialize predictor with ensemble models
    predictor = CryptoPredictor(models_dir=str(models_dir))

    # Load test data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")
    logger.info(f"Loaded {len(df)} rows of BTC data")

    # Make ensemble prediction
    ensemble_prob = predictor.predict(df)
    ensemble_class = 1 if ensemble_prob > 0.5 else 0
    confidence = max(ensemble_prob, 1 - ensemble_prob)
    direction = "UP" if ensemble_class == 1 else "DOWN"

    logger.info("\nEnsemble Model Results:")
    logger.info(f"  Models used: {list(predictor.models.keys())}")
    logger.info(f"  Average probability of price going UP: {ensemble_prob:.4f}")
    logger.info(f"  Predicted direction: {direction}")
    logger.info(f"  Confidence: {confidence:.4f}")

    return ensemble_prob, ensemble_class


def example_individual_vs_ensemble():
    """Example 3: Compare individual models (LightGBM, XGBoost) vs Ensemble."""
    logger.info("\n=== Example 3: Individual Models vs Ensemble ===")

    # Check if ensemble models exist
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("Models directory not found. Please train models first.")
        return None

    # Initialize predictor with ensemble models
    predictor = CryptoPredictor(models_dir=str(models_dir))

    # Load test data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")
    logger.info(f"Loaded {len(df)} rows of BTC data")

    # Get individual model predictions
    individual_predictions = predictor.get_individual_predictions(df)
    ensemble_prob = predictor.predict(df)

    logger.info("\nIndividual Model vs Ensemble Comparison:")
    logger.info("-" * 60)

    results = {}
    for model_name, prob in individual_predictions.items():
        direction = "UP" if prob > 0.5 else "DOWN"
        confidence = max(prob, 1 - prob)

        logger.info(
            f"{model_name.upper():>12}: {prob:.4f} ({direction}) - Confidence: {confidence:.4f}"
        )
        results[model_name] = {"prob": prob, "direction": direction, "confidence": confidence}

    # Ensemble results
    ensemble_direction = "UP" if ensemble_prob > 0.5 else "DOWN"
    ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob)
    logger.info(
        f"{'ENSEMBLE':>12}: {ensemble_prob:.4f} ({ensemble_direction}) - Confidence: {ensemble_confidence:.4f}"
    )

    results["ensemble"] = {
        "prob": ensemble_prob,
        "direction": ensemble_direction,
        "confidence": ensemble_confidence,
    }

    # Analysis
    logger.info("\nAnalysis:")
    for model_name, model_results in results.items():
        if model_name == "ensemble":
            continue

        prob_diff = abs(model_results["prob"] - ensemble_prob)
        direction_agreement = "Yes" if model_results["direction"] == ensemble_direction else "No"

        logger.info(f"  {model_name.upper()} vs Ensemble:")
        logger.info(f"    Probability difference: {prob_diff:.4f}")
        logger.info(f"    Direction agreement: {direction_agreement}")

        if prob_diff < 0.1 and direction_agreement == "Yes":
            logger.info(f"    🟢 {model_name.upper()} aligns well with ensemble")
        elif prob_diff < 0.2:
            logger.info(f"    🟡 {model_name.upper()} shows moderate agreement with ensemble")
        else:
            logger.info(f"    🔴 {model_name.upper()} shows significant disagreement with ensemble")

    return results


def compare_predictions():
    """Compare single model vs ensemble predictions."""
    logger.info("\n=== Comparing Single Model vs Ensemble ===")

    # Get predictions from both approaches
    single_prob, single_class = example_single_model_prediction()
    ensemble_prob, ensemble_class = example_ensemble_prediction()

    if single_prob is not None and ensemble_prob is not None:
        logger.info("\nComparison Summary:")
        single_direction = "UP" if single_class == 1 else "DOWN"
        ensemble_direction = "UP" if ensemble_class == 1 else "DOWN"
        logger.info(f"  Single Model (RF): {single_prob:.4f} ({single_direction})")
        logger.info(f"  Ensemble Model: {ensemble_prob:.4f} ({ensemble_direction})")
        logger.info(f"  Probability difference: {abs(single_prob - ensemble_prob):.4f}")

        # Check if they agree on direction
        agreement = "Yes" if single_class == ensemble_class else "No"
        logger.info(f"  Direction agreement: {agreement}")

        return single_prob, ensemble_prob

    return None, None


def compare_lightgbm_vs_ensemble():
    """Compare LightGBM alone vs Ensemble prediction."""
    logger.info("\n=== LightGBM vs Ensemble Comparison ===")

    # Check if ensemble models exist
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("Models directory not found. Please train models first.")
        return None, None

    # Initialize predictor
    predictor = CryptoPredictor(models_dir=str(models_dir))
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

    try:
        # Get LightGBM prediction
        lightgbm_prob = predictor.predict_single_model(df, "lightgbm")
        lightgbm_direction = "UP" if lightgbm_prob > 0.5 else "DOWN"
        lightgbm_confidence = max(lightgbm_prob, 1 - lightgbm_prob)

        # Get ensemble prediction
        ensemble_prob = predictor.predict(df)
        ensemble_direction = "UP" if ensemble_prob > 0.5 else "DOWN"
        ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob)

        logger.info("\nLightGBM vs Ensemble Results:")
        logger.info(
            f"  LightGBM: {lightgbm_prob:.4f} ({lightgbm_direction}) - Confidence: {lightgbm_confidence:.4f}"
        )
        logger.info(
            f"  Ensemble: {ensemble_prob:.4f} ({ensemble_direction}) - Confidence: {ensemble_confidence:.4f}"
        )

        prob_diff = abs(lightgbm_prob - ensemble_prob)
        direction_agreement = "Yes" if lightgbm_direction == ensemble_direction else "No"

        logger.info(f"  Probability difference: {prob_diff:.4f}")
        logger.info(f"  Direction agreement: {direction_agreement}")

        # Analysis
        if prob_diff < 0.05:
            logger.info("  🟢 LightGBM and ensemble are very similar")
        elif prob_diff < 0.15:
            logger.info("  🟡 LightGBM and ensemble show moderate agreement")
        else:
            logger.info("  🔴 LightGBM and ensemble show significant disagreement")

        return lightgbm_prob, ensemble_prob

    except ValueError as e:
        logger.error(f"Error: {e}")
        return None, None


def compare_xgboost_vs_ensemble():
    """Compare XGBoost alone vs Ensemble prediction."""
    logger.info("\n=== XGBoost vs Ensemble Comparison ===")

    # Check if ensemble models exist
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("Models directory not found. Please train models first.")
        return None, None

    # Initialize predictor
    predictor = CryptoPredictor(models_dir=str(models_dir))
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

    try:
        # Get XGBoost prediction
        xgboost_prob = predictor.predict_single_model(df, "xgboost")
        xgboost_direction = "UP" if xgboost_prob > 0.5 else "DOWN"
        xgboost_confidence = max(xgboost_prob, 1 - xgboost_prob)

        # Get ensemble prediction
        ensemble_prob = predictor.predict(df)
        ensemble_direction = "UP" if ensemble_prob > 0.5 else "DOWN"
        ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob)

        logger.info("\nXGBoost vs Ensemble Results:")
        logger.info(
            f"  XGBoost:  {xgboost_prob:.4f} ({xgboost_direction}) - Confidence: {xgboost_confidence:.4f}"
        )
        logger.info(
            f"  Ensemble: {ensemble_prob:.4f} ({ensemble_direction}) - Confidence: {ensemble_confidence:.4f}"
        )

        prob_diff = abs(xgboost_prob - ensemble_prob)
        direction_agreement = "Yes" if xgboost_direction == ensemble_direction else "No"

        logger.info(f"  Probability difference: {prob_diff:.4f}")
        logger.info(f"  Direction agreement: {direction_agreement}")

        # Analysis
        if prob_diff < 0.05:
            logger.info("  🟢 XGBoost and ensemble are very similar")
        elif prob_diff < 0.15:
            logger.info("  🟡 XGBoost and ensemble show moderate agreement")
        else:
            logger.info("  🔴 XGBoost and ensemble show significant disagreement")

        return xgboost_prob, ensemble_prob

    except ValueError as e:
        logger.error(f"Error: {e}")
        return None, None


def test_multiple_timepoints():
    """Test predictions on multiple recent timepoints."""
    logger.info("\n=== Testing Multiple Timepoints ===")

    # Load data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

    # Test on last 10 timepoints
    results = []
    for i in range(10, 0, -1):  # From 10 periods ago to current
        # Get data up to timepoint
        test_df = df.head(len(df) - i + 1)

        try:
            # Ensemble prediction
            ensemble_predictor = CryptoPredictor(models_dir="models")
            ensemble_prob = ensemble_predictor.predict(test_df)

            # Single model prediction
            single_predictor = CryptoPredictor(models_dir="models_single_rf")
            single_prob = single_predictor.predict(test_df)

            results.append(
                {
                    "timepoint": len(test_df),
                    "ensemble_prob": ensemble_prob,
                    "single_prob": single_prob,
                    "ensemble_direction": "UP" if ensemble_prob > 0.5 else "DOWN",
                    "single_direction": "UP" if single_prob > 0.5 else "DOWN",
                }
            )

        except Exception as e:
            logger.warning(f"Skipped timepoint {len(test_df)}: {e}")

    # Display results
    if results:
        logger.info(f"\nPredictions across {len(results)} timepoints:")
        logger.info(
            f"{'Timepoint':<10} {'Ensemble':<10} {'Single':<10} "
            f"{'Ens Dir':<8} {'Single Dir':<10} {'Agreement'}"
        )
        logger.info("-" * 70)

        agreements = 0
        for result in results:
            agreement = (
                "Yes" if result["ensemble_direction"] == result["single_direction"] else "No"
            )
            if agreement == "Yes":
                agreements += 1

            timepoint = result["timepoint"]
            ens_prob = result["ensemble_prob"]
            single_prob = result["single_prob"]
            ens_dir = result["ensemble_direction"]
            single_dir = result["single_direction"]
            logger.info(
                f"{timepoint:<10} {ens_prob:<10.4f} {single_prob:<10.4f} "
                f"{ens_dir:<8} {single_dir:<10} {agreement}"
            )

        agreement_rate = agreements / len(results) * 100
        logger.info(f"\nAgreement rate: {agreement_rate:.1f}% ({agreements}/{len(results)})")

    return results


def analyze_model_confidence():
    """Analyze confidence levels of predictions."""
    logger.info("\n=== Analyzing Model Confidence ===")

    # Load data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

    # Make predictions using both models
    ensemble_predictor = CryptoPredictor(models_dir="models")
    single_predictor = CryptoPredictor(models_dir="models_single_rf")

    ensemble_prob = ensemble_predictor.predict(df)
    single_prob = single_predictor.predict(df)

    # Calculate confidence (distance from 0.5)
    ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob)
    single_confidence = max(single_prob, 1 - single_prob)

    logger.info("Confidence Analysis:")
    logger.info(f"  Ensemble confidence: {ensemble_confidence:.4f}")
    logger.info(f"  Single model confidence: {single_confidence:.4f}")

    # Confidence categories
    def get_confidence_category(confidence):
        if confidence > 0.8:
            return "High"
        elif confidence > 0.6:
            return "Medium"
        else:
            return "Low"

    logger.info(f"  Ensemble confidence level: {get_confidence_category(ensemble_confidence)}")
    logger.info(f"  Single model confidence level: {get_confidence_category(single_confidence)}")

    return ensemble_confidence, single_confidence


def main():
    """Run all examples."""
    logger.info("Starting cryptocurrency prediction examples...")
    logger.info("=" * 60)

    try:
        # Example 1: Single model prediction
        single_prob, single_class = example_single_model_prediction()

        # Example 2: Ensemble prediction
        ensemble_prob, ensemble_class = example_ensemble_prediction()

        # Example 3: Individual models vs ensemble
        example_individual_vs_ensemble()

        # Compare LightGBM vs Ensemble
        compare_lightgbm_vs_ensemble()

        # Compare XGBoost vs Ensemble
        compare_xgboost_vs_ensemble()

        # Compare predictions
        compare_predictions()

        # Test multiple timepoints
        test_multiple_timepoints()

        # Analyze confidence
        analyze_model_confidence()

        logger.info("\n" + "=" * 60)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        # Final summary
        if single_prob is not None and ensemble_prob is not None:
            logger.info("\nFinal Summary:")
            single_dir = "UP" if single_class == 1 else "DOWN"
            ensemble_dir = "UP" if ensemble_class == 1 else "DOWN"
            logger.info(f"  Single Model (Random Forest): {single_prob:.4f} ({single_dir})")
            logger.info(
                f"  Ensemble Model (LightGBM + XGBoost): {ensemble_prob:.4f} ({ensemble_dir})"
            )

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
