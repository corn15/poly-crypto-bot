# Cryptocurrency Prediction Examples

This directory contains updated examples demonstrating how to use the latest version of the cryptocurrency price prediction system.

## Overview

The examples have been updated to work with the simplified `predictor.py` interface, which now provides a clean, easy-to-use API for making predictions with trained models. The examples now include comprehensive comparisons between individual models (LightGBM or XGBoost) and ensemble predictions.

## Files

### Core Files
- **`example_usage.py`** - Comprehensive examples with detailed analysis
- **`simple_prediction_test.py`** - Clean, focused examples (recommended for beginners)
- **`demo_predictions.py`** - Quick demo script to test everything works
- **`individual_vs_ensemble_comparison.py`** - Focused script for comparing individual models vs ensemble

### Model Directories
- **`models/`** - Contains ensemble models (LightGBM + XGBoost)
- **`models_single_rf/`** - Contains single Random Forest model (created by examples)
- **`models_single_example/`** - Contains single model for simple example

## Quick Start

### 1. Run Simple Examples
```bash
python simple_prediction_test.py
```

This script demonstrates:
- **Example 1**: Train and use a single Random Forest model
- **Example 2**: Use existing ensemble models (LightGBM + XGBoost)
- **Comparison**: Compare predictions from both approaches

### 2. Run Comprehensive Demo
```bash
python demo_predictions.py
```

This runs all available examples and provides detailed analysis.

### 3. Run Full Example Suite
```bash
python example_usage.py
```

This provides the most comprehensive set of examples with advanced features.

### 4. Run Individual vs Ensemble Comparison
```bash
# Compare all individual models with ensemble
python individual_vs_ensemble_comparison.py

# Compare specific model with ensemble
python individual_vs_ensemble_comparison.py --model lightgbm
python individual_vs_ensemble_comparison.py --model xgboost

# Include time series analysis
python individual_vs_ensemble_comparison.py --model xgboost --time-series
```

## Examples Explained

### Example 1: Single Model Prediction

Creates and trains a single Random Forest model, then uses it for prediction:

```python
from src.prediction.predictor import CryptoPredictor
import polars as pl

# Load data
df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

# Use single model
predictor = CryptoPredictor(models_dir="models_single_example")
probability = predictor.predict(df)

print(f"Probability of price going UP: {probability:.4f}")
print(f"Prediction: {'BUY' if probability > 0.5 else 'SELL'}")
```

**Key Features:**
- Trains a Random Forest model from scratch
- Saves model artifacts (model, scaler, features)
- Demonstrates the complete workflow from training to prediction

### Example 2: Ensemble Prediction

Uses existing pre-trained LightGBM and XGBoost models:

```python
from src.prediction.predictor import CryptoPredictor
import polars as pl

# Load data
df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

# Use ensemble models
predictor = CryptoPredictor(models_dir="models")
probability = predictor.predict(df)

print(f"Average probability from ensemble: {probability:.4f}")
print(f"Models used: {list(predictor.models.keys())}")
```

**Key Features:**
- Uses multiple pre-trained models
- Automatically averages predictions from all models
- Higher accuracy through ensemble learning

### Example 3: Individual Model vs Ensemble Comparison

Compares individual models (LightGBM or XGBoost) with ensemble predictions:

```python
from src.prediction.predictor import CryptoPredictor
import polars as pl

# Load data
df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

# Use ensemble predictor
predictor = CryptoPredictor(models_dir="models")

# Get individual model predictions
lightgbm_prob = predictor.predict_single_model(df, "lightgbm")
xgboost_prob = predictor.predict_single_model(df, "xgboost")
ensemble_prob = predictor.predict(df)

print(f"LightGBM: {lightgbm_prob:.4f}")
print(f"XGBoost:  {xgboost_prob:.4f}")
print(f"Ensemble: {ensemble_prob:.4f}")
```

**Key Features:**
- Compare individual models against ensemble
- Analyze model agreement and disagreement
- Understand contribution of each model to ensemble
- Time series analysis across multiple timepoints

## Current Predictor API

The updated `predictor.py` has a simplified interface:

```python
class CryptoPredictor:
    def __init__(self, models_dir: str = "models"):
        # Loads all models, scaler, and feature names from directory
        
    def predict(self, df: pl.DataFrame) -> float:
        # Returns ensemble prediction probability (0.0 to 1.0)
        # Uses the most recent data point from the dataframe
        
    def predict_single_model(self, df: pl.DataFrame, model_name: str) -> float:
        # Returns prediction from a specific model
        
    def get_individual_predictions(self, df: pl.DataFrame) -> dict:
        # Returns predictions from all individual models
```

## Model Directory Structure

Each model directory should contain:
- `*_model.joblib` - Trained model files
- `scaler.joblib` - Feature scaler
- `features.json` - List of feature names used during training

Example structure:
```
models/
├── lightgbm_model.joblib
├── xgboost_model.joblib
├── scaler.joblib
└── features.json
```

## Understanding the Output

### Prediction Values
- **0.0 to 1.0**: Probability that the price will go UP
- **> 0.5**: Suggests BUY signal
- **< 0.5**: Suggests SELL signal

### Confidence Levels
- **High (> 0.8)**: Strong confidence in the prediction
- **Medium (0.6 - 0.8)**: Moderate confidence
- **Low (< 0.6)**: Weak confidence, use caution

## Example Output

```
Single Model Results:
  Probability of price going UP: 0.9800
  Prediction: BUY
  Confidence: 0.9800

Ensemble Model Results:
  Average probability of price going UP: 0.2814
  Prediction: SELL
  Confidence: 0.7186

Individual Model Comparisons:
  LIGHTGBM: 0.2066 (SELL) - Confidence: 0.7934
   XGBOOST: 0.3562 (SELL) - Confidence: 0.6438

LightGBM vs Ensemble:
  Difference: 0.0748
  Agreement: YES
  🟡 LIGHTGBM shows good agreement with ensemble

XGBoost vs Ensemble:
  Difference: 0.0748
  Agreement: YES
  🟡 XGBOOST shows good agreement with ensemble
```

## Tips for Usage

1. **Use ensemble models** when available for better accuracy
2. **Check confidence levels** before making trading decisions
3. **Compare individual models** to understand ensemble composition
4. **Monitor model agreement** - disagreement may indicate uncertainty
5. **Analyze time series patterns** to validate model stability
6. **Consider additional factors** beyond model predictions
7. **Test on historical data** before using in production

## Requirements

- Python 3.8+
- Required packages: `polars`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `joblib`
- Training data in `data/` directory
- Trained models (run training scripts first if models don't exist)

## Troubleshooting

### "No models found" Error
- Ensure the models directory exists and contains `*_model.joblib` files
- Run the training script to create models
- Check the `models_dir` parameter in `CryptoPredictor()`

### "Feature names not found" Error
- Ensure `features.json` exists in the models directory
- Check that the features file contains a valid JSON list

### Import Errors
- Install required packages: `pip install polars numpy scikit-learn lightgbm xgboost joblib`
- Ensure you're in the project root directory

## Advanced Usage

### Individual Model Analysis
```bash
# Compare LightGBM specifically
python individual_vs_ensemble_comparison.py --model lightgbm

# Compare XGBoost with time series analysis
python individual_vs_ensemble_comparison.py --model xgboost --time-series --num-points 20
```

### Understanding Model Contributions
- **LightGBM vs Ensemble**: Shows how LightGBM pulls the ensemble prediction
- **XGBoost vs Ensemble**: Shows XGBoost's influence on final prediction
- **Model Spread Analysis**: Measures disagreement between individual models
- **Time Series Stability**: Tracks prediction consistency over time

## Next Steps

- Explore different model architectures
- Implement real-time prediction endpoints
- Add backtesting functionality
- Create automated trading signals
- Integrate with live market data feeds
- Analyze model performance under different market conditions
- Implement dynamic model weighting based on recent performance