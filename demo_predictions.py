#!/usr/bin/env python3
"""
Simple demo script to test the updated example_usage.py with the latest predictor.py

This script demonstrates both single model and ensemble prediction capabilities.
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_requirements():
    """Check if required files exist."""
    required_files = [
        "data/BTCUSDT_5m_20240901.parquet",
        "models/lightgbm_model.joblib",
        "models/xgboost_model.joblib",
        "models/scaler.joblib",
        "models/features.json",
    ]

    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)

    if missing_files:
        logger.warning("Missing files:")
        for file_path in missing_files:
            logger.warning(f"  - {file_path}")
        logger.warning("Some examples may not work properly.")
        return False

    logger.info("All required files found!")
    return True


def main():
    """Run the prediction demos."""
    logger.info("=" * 60)
    logger.info("CRYPTOCURRENCY PREDICTION DEMO")
    logger.info("=" * 60)

    # Check requirements
    check_requirements()

    try:
        # Import and run examples
        from example_usage import (
            analyze_model_confidence,
            compare_lightgbm_vs_ensemble,
            compare_predictions,
            compare_xgboost_vs_ensemble,
            example_ensemble_prediction,
            example_individual_vs_ensemble,
            example_single_model_prediction,
            test_multiple_timepoints,
        )

        logger.info("\nStarting prediction demos...")

        # Run single model example
        logger.info("\n" + "=" * 40)
        logger.info("DEMO 1: SINGLE MODEL PREDICTION")
        logger.info("=" * 40)
        single_prob, single_class = example_single_model_prediction()

        # Run ensemble example
        logger.info("\n" + "=" * 40)
        logger.info("DEMO 2: ENSEMBLE PREDICTION")
        logger.info("=" * 40)
        ensemble_prob, ensemble_class = example_ensemble_prediction()

        # Compare results
        if single_prob is not None and ensemble_prob is not None:
            logger.info("\n" + "=" * 40)
            logger.info("DEMO 3: INDIVIDUAL vs ENSEMBLE")
            logger.info("=" * 40)
            example_individual_vs_ensemble()

            logger.info("\n" + "=" * 40)
            logger.info("DEMO 4: LIGHTGBM vs ENSEMBLE")
            logger.info("=" * 40)
            compare_lightgbm_vs_ensemble()

            logger.info("\n" + "=" * 40)
            logger.info("DEMO 5: XGBOOST vs ENSEMBLE")
            logger.info("=" * 40)
            compare_xgboost_vs_ensemble()

            logger.info("\n" + "=" * 40)
            logger.info("DEMO 6: SINGLE vs ENSEMBLE COMPARISON")
            logger.info("=" * 40)
            compare_predictions()

            # Additional analysis
            logger.info("\n" + "=" * 40)
            logger.info("DEMO 7: CONFIDENCE ANALYSIS")
            logger.info("=" * 40)
            analyze_model_confidence()

        logger.info("\n" + "=" * 60)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)

        # Summary
        if single_prob is not None and ensemble_prob is not None:
            logger.info("\nQUICK SUMMARY:")
            logger.info(
                f"Single Model Prediction: {single_prob:.4f} ({'BUY' if single_class == 1 else 'SELL'})"
            )
            logger.info(
                f"Ensemble Prediction: {ensemble_prob:.4f} ({'BUY' if ensemble_class == 1 else 'SELL'})"
            )
            logger.info(f"Models Agree: {'YES' if single_class == ensemble_class else 'NO'}")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure example_usage.py is updated and all dependencies are installed.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        logger.error("Check that data files exist and models are trained.")
        sys.exit(1)


if __name__ == "__main__":
    main()
