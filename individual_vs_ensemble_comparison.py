#!/usr/bin/env python3
"""
Individual vs Ensemble Model Comparison Script

This script focuses specifically on comparing individual models (LightGBM or XGBoost)
with ensemble predictions to understand their contributions and disagreements.

Usage:
    python individual_vs_ensemble_comparison.py
    python individual_vs_ensemble_comparison.py --model lightgbm
    python individual_vs_ensemble_comparison.py --model xgboost
"""

import argparse
import logging
from pathlib import Path

import polars as pl

from src.prediction.predictor import CryptoPredictor

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compare_individual_vs_ensemble(
    model_name: str = None, data_file: str = "data/BTCUSDT_5m_20240901.parquet"
):
    """Compare individual model predictions with ensemble predictions."""

    logger.info("=" * 80)
    logger.info("INDIVIDUAL vs ENSEMBLE MODEL COMPARISON")
    logger.info("=" * 80)

    # Check if models exist
    models_dir = Path("models")
    if not models_dir.exists():
        logger.error("Models directory not found. Please train models first.")
        return

    # Initialize predictor
    predictor = CryptoPredictor(models_dir=str(models_dir))
    available_models = list(predictor.models.keys())

    logger.info(f"Available models: {available_models}")

    if model_name and model_name not in available_models:
        logger.error(f"Model '{model_name}' not found. Available: {available_models}")
        return

    # Load data
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        return

    df = pl.read_parquet(data_file)
    logger.info(f"Loaded {len(df)} rows from {data_file}")

    # Get ensemble prediction
    ensemble_prob = predictor.predict(df)
    ensemble_direction = "BUY" if ensemble_prob > 0.5 else "SELL"
    ensemble_confidence = max(ensemble_prob, 1 - ensemble_prob)

    logger.info("\nENSEMBLE PREDICTION:")
    logger.info(f"  Probability: {ensemble_prob:.4f}")
    logger.info(f"  Direction:   {ensemble_direction}")
    logger.info(f"  Confidence:  {ensemble_confidence:.4f}")

    # Compare with individual models
    if model_name:
        # Compare specific model
        compare_single_model(
            predictor, df, model_name, ensemble_prob, ensemble_direction, ensemble_confidence
        )
    else:
        # Compare all individual models
        individual_predictions = predictor.get_individual_predictions(df)

        logger.info("\nINDIVIDUAL MODEL COMPARISONS:")
        logger.info("-" * 60)

        for model_name, prob in individual_predictions.items():
            compare_single_model(
                predictor,
                df,
                model_name,
                ensemble_prob,
                ensemble_direction,
                ensemble_confidence,
                prob,
            )

        # Additional analysis
        analyze_model_contributions(individual_predictions, ensemble_prob)


def compare_single_model(
    predictor,
    df,
    model_name,
    ensemble_prob,
    ensemble_direction,
    ensemble_confidence,
    individual_prob=None,
):
    """Compare a single model with ensemble."""

    if individual_prob is None:
        individual_prob = predictor.predict_single_model(df, model_name)

    individual_direction = "BUY" if individual_prob > 0.5 else "SELL"
    individual_confidence = max(individual_prob, 1 - individual_prob)

    # Calculate differences
    prob_diff = abs(individual_prob - ensemble_prob)
    direction_agreement = "YES" if individual_direction == ensemble_direction else "NO"

    logger.info(f"\n{model_name.upper()} vs ENSEMBLE:")
    logger.info(
        f"  {model_name.upper():>10} Probability: {individual_prob:.4f} ({individual_direction})"
    )
    logger.info(f"  {'ENSEMBLE':>10} Probability: {ensemble_prob:.4f} ({ensemble_direction})")
    logger.info(f"  Difference:        {prob_diff:.4f}")
    logger.info(f"  Direction Agreement: {direction_agreement}")
    logger.info(f"  {model_name.upper()} Confidence:   {individual_confidence:.4f}")
    logger.info(f"  Ensemble Confidence: {ensemble_confidence:.4f}")

    # Analysis
    if prob_diff < 0.05:
        logger.info(f"  🟢 {model_name.upper()} is very close to ensemble (difference < 5%)")
    elif prob_diff < 0.10:
        logger.info(
            f"  🟡 {model_name.upper()} shows good agreement with ensemble (difference < 10%)"
        )
    elif prob_diff < 0.20:
        logger.info(f"  🟠 {model_name.upper()} shows moderate disagreement (difference < 20%)")
    else:
        logger.info(f"  🔴 {model_name.upper()} shows significant disagreement (difference ≥ 20%)")

    # Confidence analysis
    if individual_confidence > 0.8 and ensemble_confidence > 0.8:
        if direction_agreement == "YES":
            logger.info(
                f"  ✅ Both {model_name.upper()} and ensemble are highly confident and agree"
            )
        else:
            logger.info(
                f"  ⚠️  Both {model_name.upper()} and ensemble are highly confident but DISAGREE!"
            )
    elif individual_confidence > 0.7 or ensemble_confidence > 0.7:
        logger.info("  ℹ️  Moderate confidence levels - consider additional analysis")
    else:
        logger.info("  ⚠️  Low confidence levels - predictions may be unreliable")


def analyze_model_contributions(individual_predictions, ensemble_prob):
    """Analyze how individual models contribute to the ensemble."""

    logger.info("\nMODEL CONTRIBUTION ANALYSIS:")
    logger.info("-" * 40)

    total_models = len(individual_predictions)

    for model_name, prob in individual_predictions.items():
        # Calculate how much this model "pulls" the ensemble toward its prediction
        contribution_direction = "higher" if prob > ensemble_prob else "lower"
        contribution_magnitude = abs(prob - ensemble_prob)

        logger.info(f"{model_name.upper()}:")
        logger.info(f"  Individual prediction: {prob:.4f}")
        logger.info(f"  Pulls ensemble {contribution_direction} by: {contribution_magnitude:.4f}")
        logger.info(
            f"  Relative influence: {contribution_magnitude / sum(abs(p - ensemble_prob) for p in individual_predictions.values()) * 100:.1f}%"
        )

    # Overall ensemble characteristics
    model_spread = max(individual_predictions.values()) - min(individual_predictions.values())
    logger.info("\nENSEMBLE CHARACTERISTICS:")
    logger.info(f"  Model spread (max - min): {model_spread:.4f}")
    logger.info(
        f"  Average individual prediction: {sum(individual_predictions.values()) / len(individual_predictions):.4f}"
    )
    logger.info(f"  Ensemble prediction: {ensemble_prob:.4f}")

    if model_spread < 0.10:
        logger.info("  🟢 Low model disagreement - ensemble is stable")
    elif model_spread < 0.30:
        logger.info("  🟡 Moderate model disagreement - ensemble provides balance")
    else:
        logger.info("  🔴 High model disagreement - ensemble may mask important signals")


def test_multiple_timepoints(model_name: str = None, num_points: int = 10):
    """Test individual vs ensemble predictions across multiple timepoints."""

    logger.info(f"\nTIME SERIES ANALYSIS - Last {num_points} timepoints:")
    logger.info("=" * 60)

    # Load data
    df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")
    predictor = CryptoPredictor(models_dir="models")

    results = []

    for i in range(num_points, 0, -1):
        # Get data up to this timepoint
        test_df = df.head(len(df) - i + 1)

        try:
            # Get ensemble prediction
            ensemble_prob = predictor.predict(test_df)
            ensemble_direction = "BUY" if ensemble_prob > 0.5 else "SELL"

            if model_name:
                # Get specific model prediction
                individual_prob = predictor.predict_single_model(test_df, model_name)
                individual_direction = "BUY" if individual_prob > 0.5 else "SELL"

                agreement = "YES" if individual_direction == ensemble_direction else "NO"
                prob_diff = abs(individual_prob - ensemble_prob)

                results.append(
                    {
                        "timepoint": len(test_df),
                        "individual_prob": individual_prob,
                        "ensemble_prob": ensemble_prob,
                        "prob_diff": prob_diff,
                        "agreement": agreement,
                    }
                )
            else:
                # Get all individual predictions
                individual_preds = predictor.get_individual_predictions(test_df)
                results.append(
                    {
                        "timepoint": len(test_df),
                        "ensemble_prob": ensemble_prob,
                        "individual_preds": individual_preds,
                    }
                )

        except Exception as e:
            logger.warning(f"Skipped timepoint {len(test_df)}: {e}")

    # Display results
    if model_name and results:
        logger.info(f"\n{model_name.upper()} vs ENSEMBLE - Time Series:")
        logger.info(
            f"{'Timepoint':<10} {model_name.upper():<10} {'Ensemble':<10} {'Diff':<8} {'Agreement'}"
        )
        logger.info("-" * 55)

        total_agreement = 0
        for result in results:
            if result["agreement"] == "YES":
                total_agreement += 1
            logger.info(
                f"{result['timepoint']:<10} {result['individual_prob']:<10.4f} "
                f"{result['ensemble_prob']:<10.4f} {result['prob_diff']:<8.4f} {result['agreement']}"
            )

        agreement_rate = total_agreement / len(results) * 100
        avg_diff = sum(r["prob_diff"] for r in results) / len(results)

        logger.info("\nSUMMARY:")
        logger.info(f"  Agreement rate: {agreement_rate:.1f}% ({total_agreement}/{len(results)})")
        logger.info(f"  Average probability difference: {avg_diff:.4f}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Compare individual models with ensemble predictions"
    )
    parser.add_argument(
        "--model",
        choices=["lightgbm", "xgboost"],
        help="Specific model to compare (if not specified, compares all)",
    )
    parser.add_argument(
        "--data", default="data/BTCUSDT_5m_20240901.parquet", help="Path to data file"
    )
    parser.add_argument("--time-series", action="store_true", help="Run time series analysis")
    parser.add_argument(
        "--num-points", type=int, default=10, help="Number of timepoints for time series analysis"
    )

    args = parser.parse_args()

    try:
        # Main comparison
        compare_individual_vs_ensemble(args.model, args.data)

        # Time series analysis if requested
        if args.time_series:
            test_multiple_timepoints(args.model, args.num_points)

        logger.info("\n" + "=" * 80)
        logger.info("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error("Make sure you have trained models and data files available.")


if __name__ == "__main__":
    main()
