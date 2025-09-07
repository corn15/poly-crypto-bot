#!/usr/bin/env python3
"""
Example script demonstrating how to use the cryptocurrency price prediction models.

This script shows various usage patterns including:
- Loading and using trained models
- Making single predictions
- Making batch predictions
- Analyzing feature importance
- Working with different data formats
"""

import logging
from pathlib import Path

import numpy as np
import polars as pl

from src.prediction.predictor import CryptoPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def example_single_prediction():
    """Example: Make a single prediction on the latest data point."""
    logger.info("=== Example 1: Single Prediction ===")

    # Initialize predictor
    predictor = CryptoPredictor(models_dir="models")

    # Load sample data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")
    logger.info(f"Loaded {len(df)} rows of BTC data")

    # Make single prediction (uses latest data point)
    predictions = predictor.predict_single(df)

    # Print results
    predictor.print_predictions(predictions)

    # Get ensemble prediction
    ensemble = predictor.get_ensemble_prediction(predictions)
    logger.info(f"\nEnsemble prediction: {ensemble}")

    return predictions, ensemble


def example_batch_predictions():
    """Example: Make batch predictions on all valid data points."""
    logger.info("\n=== Example 2: Batch Predictions ===")

    # Initialize predictor
    predictor = CryptoPredictor(models_dir="models")

    # Load sample data (using a smaller subset for demo)
    df = pl.read_parquet("data/ETHUSDT_5m_20240901.parquet").head(1000)
    logger.info(f"Loaded {len(df)} rows of ETH data for batch prediction")

    # Make batch predictions
    predictions = predictor.predict_batch(df)

    # Analyze results
    for model_name, pred_data in predictions.items():
        probs = pred_data["probabilities"]
        preds = pred_data["predictions"]

        logger.info(f"\n{model_name.upper()} Batch Results:")
        logger.info(f"  Total predictions: {len(probs)}")
        logger.info(f"  Average probability: {np.mean(probs):.4f}")
        logger.info(f"  Probability std: {np.std(probs):.4f}")
        logger.info(f"  Predictions UP: {np.sum(preds)} ({np.mean(preds):.1%})")
        logger.info(
            f"  High confidence predictions (>0.7): {np.sum(np.maximum(probs, 1 - probs) > 0.7)}"
        )

    return predictions


def example_compare_cryptocurrencies():
    """Example: Compare predictions across different cryptocurrencies."""
    logger.info("\n=== Example 3: Compare Cryptocurrencies ===")

    # Initialize predictor
    predictor = CryptoPredictor(models_dir="models")

    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    results = {}

    for symbol in symbols:
        data_file = f"data/{symbol}_5m_20240901.parquet"
        if Path(data_file).exists():
            logger.info(f"\nAnalyzing {symbol}...")

            # Load data
            df = pl.read_parquet(data_file)

            # Make prediction on latest data
            predictions = predictor.predict_single(df)
            ensemble = predictor.get_ensemble_prediction(predictions)

            results[symbol] = {
                "ensemble_prob": ensemble["probability"],
                "ensemble_pred": ensemble["prediction"],
                "lightgbm_prob": predictions.get("lightgbm", {}).get("probability", 0),
                "xgboost_prob": predictions.get("xgboost", {}).get("probability", 0),
            }

            direction = "UP" if ensemble["prediction"] == 1 else "DOWN"
            confidence = max(ensemble["probability"], 1 - ensemble["probability"])

            logger.info(f"  {symbol}: {direction} (confidence: {confidence:.3f})")

    # Summary comparison
    logger.info("\n" + "=" * 60)
    logger.info("CRYPTOCURRENCY COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"{'Symbol':<10} {'Direction':<10} {'Confidence':<12} {'LGB Prob':<10} {'XGB Prob':<10}"
    )
    logger.info("-" * 60)

    for symbol, result in results.items():
        direction = "UP" if result["ensemble_pred"] == 1 else "DOWN"
        confidence = max(result["ensemble_prob"], 1 - result["ensemble_prob"])

        logger.info(
            f"{symbol:<10} {direction:<10} {confidence:<12.3f} "
            f"{result['lightgbm_prob']:<10.3f} {result['xgboost_prob']:<10.3f}"
        )

    return results


def example_confidence_analysis():
    """Example: Analyze prediction confidence levels."""
    logger.info("\n=== Example 4: Confidence Analysis ===")

    # Initialize predictor
    predictor = CryptoPredictor(models_dir="models")

    # Load data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet").head(5000)  # Use subset for speed

    # Make batch predictions
    predictions = predictor.predict_batch(df)

    # Analyze confidence levels
    for model_name, pred_data in predictions.items():
        probs = pred_data["probabilities"]

        # Calculate confidence (distance from 0.5)
        confidence = np.maximum(probs, 1 - probs)

        # Analyze confidence distribution
        high_conf = np.sum(confidence > 0.8)
        med_conf = np.sum((confidence > 0.6) & (confidence <= 0.8))
        low_conf = np.sum(confidence <= 0.6)

        logger.info(f"\n{model_name.upper()} Confidence Analysis:")
        logger.info(f"  High confidence (>0.8): {high_conf} ({high_conf / len(probs):.1%})")
        logger.info(f"  Medium confidence (0.6-0.8): {med_conf} ({med_conf / len(probs):.1%})")
        logger.info(f"  Low confidence (≤0.6): {low_conf} ({low_conf / len(probs):.1%})")
        logger.info(f"  Average confidence: {np.mean(confidence):.3f}")
        logger.info(f"  Max confidence: {np.max(confidence):.3f}")
        logger.info(f"  Min confidence: {np.min(confidence):.3f}")


def example_feature_analysis():
    """Example: Analyze which features are most important."""
    logger.info("\n=== Example 5: Feature Importance Analysis ===")

    # Load results from training
    results_file = Path("models/results.json")
    if results_file.exists():
        import json

        with open(results_file, "r") as f:
            results = json.load(f)

        # Display feature importance for both models
        for model in ["lightgbm", "xgboost"]:
            importance_key = f"{model}_feature_importance"
            if importance_key in results:
                logger.info(f"\n{model.upper()} Feature Importance:")

                # Sort features by importance
                importance_dict = results[importance_key]
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

                # Display top 10 features
                for i, (feature, importance) in enumerate(sorted_features[:10], 1):
                    logger.info(f"  {i:2d}. {feature:<25}: {importance:.4f}")
    else:
        logger.warning("Results file not found. Run train_model.py first.")


def example_custom_data():
    """Example: Use the predictor with custom data format."""
    logger.info("\n=== Example 6: Custom Data Format ===")

    # Create sample custom data (simulating real-time data feed)
    logger.info("Creating sample custom data...")

    # Load base data and take last 100 rows as "custom" data
    base_df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet").tail(100)

    # Simulate adding some noise/variation
    np.random.seed(42)
    base_df = base_df.with_columns(
        [
            (pl.col("close") * (1 + np.random.normal(0, 0.001, len(base_df)))).alias("close"),
            (pl.col("high") * (1 + np.random.normal(0, 0.001, len(base_df)))).alias("high"),
            (pl.col("low") * (1 + np.random.normal(0, 0.001, len(base_df)))).alias("low"),
        ]
    )

    logger.info(f"Custom data shape: {base_df.shape}")

    # Initialize predictor
    predictor = CryptoPredictor(models_dir="models")

    # Make prediction
    predictions = predictor.predict_single(base_df)
    ensemble = predictor.get_ensemble_prediction(predictions)

    logger.info(f"Custom data prediction: {ensemble}")

    return predictions


def main():
    """Run all examples."""
    logger.info("Starting cryptocurrency prediction examples...")

    try:
        # Example 1: Single prediction
        single_pred, ensemble = example_single_prediction()

        # Example 2: Batch predictions
        batch_pred = example_batch_predictions()

        # Example 3: Compare cryptocurrencies
        comparison = example_compare_cryptocurrencies()

        # Example 4: Confidence analysis
        example_confidence_analysis()

        # Example 5: Feature importance
        example_feature_analysis()

        # Example 6: Custom data
        custom_pred = example_custom_data()

        logger.info("\n" + "=" * 60)
        logger.info("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()
