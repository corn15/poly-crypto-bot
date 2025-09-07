# Cryptocurrency Hourly Price Direction Prediction

A machine learning system that predicts the probability that the closing price of the current hour will be higher than its opening price, using 5-minute candlestick data from the past 3 hours.

## 🎯 Objective

Predict the probability (ranging from 0 to 1) that the closing price of the current hour will be higher than its opening price, based on historical 5-minute candlestick data.

## 📊 Data

The system uses 5-minute candlestick data including:
- Open price
- Close price  
- High price
- Low price
- Volume (when available)

**Sample data files:**
- `data/BTCUSDT_5m_20240901.parquet`
- `data/ETHUSDT_5m_20240901.parquet` 
- `data/SOLUSDT_5m_20240901.parquet`

Data processing is performed using **Polars** for high-performance data manipulation.

## 🔧 Features

The system generates 24 technical features:

### Time-based Features
- `minute_in_hour`: Proportion of minutes passed within the current hour (0 to 1)

### Return Features
- `ret_5m`: Log return over the last 5 minutes
- `ret_15m`: Log return over the last 15 minutes
- `ret_30m`: Log return over the last 30 minutes
- `ret_1h`: Log return over the last 1 hour
- `ret_2h`: Log return over the last 2 hours
- `ret_3h`: Log return over the last 3 hours

### Price Position Features
- `price_deviation_from_open`: (Current price - Hourly open price) / Hourly open price
- `relative_price_position`: (Current price - Hourly low) / (Hourly high - Hourly low)

### Volume Features
- `vol_share_in_hour`: Current hour's accumulated volume / Average hourly volume over the past 3 hours

### EMA Ratio Features
- `em5_ratio`: Current price / EMA-5
- `em10_ratio`: Current price / EMA-10
- `em20_ratio`: Current price / EMA-20
- `em30_ratio`: Current price / EMA-30
- `ema5_ema10_ratio`: EMA-5 / EMA-10
- `ema5_ema20_ratio`: EMA-5 / EMA-20
- `ema5_ema30_ratio`: EMA-5 / EMA-30
- `ema10_ema20_ratio`: EMA-10 / EMA-20
- `ema10_ema30_ratio`: EMA-10 / EMA-30
- `ema20_ema30_ratio`: EMA-20 / EMA-30

### RSI Features
- `rsi_5`: RSI with a 5-period window
- `rsi_10`: RSI with a 10-period window
- `rsi_20`: RSI with a 20-period window
- `rsi_30`: RSI with a 30-period window

## 🤖 Models

The system uses two gradient boosting models:
- **LightGBM**: Fast and efficient gradient boosting
- **XGBoost**: Robust gradient boosting with excellent performance

Both models are trained with binary classification objective and provide probability outputs.

## 📈 Performance

Latest training results on combined dataset (BTC, ETH, SOL):

| Model | Accuracy | ROC AUC | Log Loss |
|-------|----------|---------|----------|
| LightGBM | 80.72% | 0.8938 | 0.4155 |
| XGBoost | 82.65% | 0.9131 | 0.3707 |
| Ensemble | - | - | - |

## 🚀 Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd polymarket-crypto-expiry-prediction
```

2. **Install dependencies using uv:**
```bash
uv sync
```

Or manually install:
```bash
pip install polars lightgbm xgboost scikit-learn pandas matplotlib seaborn
```

## 📋 Usage

### Training Models

Train models on the provided data:

```bash
python train_model.py
```

This will:
- Load data from all available cryptocurrency pairs
- Generate 24 technical features
- Train LightGBM and XGBoost models
- Evaluate performance on test set
- Save models and artifacts to `models/` directory

### Making Predictions

#### Single Prediction (Latest Data Point)
```bash
python predict.py --data data/BTCUSDT_5m_20240901.parquet --mode single
```

#### Batch Predictions (All Data Points)
```bash
python predict.py --data data/BTCUSDT_5m_20240901.parquet --mode batch
```

### Python API Usage

```python
from predict import CryptoPredictor
import polars as pl

# Initialize predictor
predictor = CryptoPredictor(models_dir="models")

# Load your data
df = pl.read_parquet("data/BTCUSDT_5m_20240901.parquet")

# Make single prediction (latest data)
predictions = predictor.predict_single(df)

# Print results
predictor.print_predictions(predictions)

# Get ensemble prediction
ensemble = predictor.get_ensemble_prediction(predictions)
print(f"Ensemble probability: {ensemble['probability']:.4f}")
```

## 📁 Project Structure

```
polymarket-crypto-expiry-prediction/
├── data/                          # Data directory
│   ├── BTCUSDT_5m_20240901.parquet
│   ├── ETHUSDT_5m_20240901.parquet
│   ├── SOLUSDT_5m_20240901.parquet
│   └── analyze_example.py
├── models/                        # Saved models and artifacts
│   ├── lightgbm_model.joblib
│   ├── xgboost_model.joblib
│   ├── scaler.joblib
│   ├── features.json
│   └── results.json
├── src/                          # Source code
├── train_model.py                # Main training script
├── predict.py                    # Inference script
├── pyproject.toml               # Dependencies
└── README.md                    # This file
```

## 🔍 Feature Importance

Based on the latest training, the most important features are:

### LightGBM Top Features:
1. `relative_price_position` - Price position within hourly range
2. `minute_in_hour` - Time within current hour
3. `price_deviation_from_open` - Deviation from hourly open
4. `ret_3h` - 3-hour return
5. `ret_1h` - 1-hour return

### XGBoost Top Features:
1. `ret_2h` - 2-hour return
2. `minute_in_hour` - Time within current hour
3. `rsi_30` - 30-period RSI
4. `ret_3h` - 3-hour return
5. `price_deviation_from_open` - Deviation from hourly open

## ⚙️ Configuration

### Model Hyperparameters

**LightGBM:**
- `objective`: binary
- `num_leaves`: 31
- `learning_rate`: 0.05
- `feature_fraction`: 0.9
- `bagging_fraction`: 0.8

**XGBoost:**
- `objective`: binary:logistic
- `learning_rate`: 0.05
- `max_depth`: 6
- `n_estimators`: 1000
- `subsample`: 0.8

### Data Preprocessing
- Features are standardized using `StandardScaler`
- Missing values are handled with forward fill and zero imputation
- Infinite values are clipped to reasonable bounds

## 📊 Data Requirements

Input data should contain columns:
- `open_datetime` or `open_time`: Timestamp
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume (optional)

## 🎯 Target Definition

The target variable is binary:
- **1**: Hourly closing price > hourly opening price (UP)
- **0**: Hourly closing price ≤ hourly opening price (DOWN)

The model outputs probabilities between 0 and 1, representing the likelihood of an UP movement.

## 🚨 Important Notes

1. **Data Dependency**: The model requires at least 3 hours of historical data to generate all features
2. **Time Alignment**: Ensure data timestamps are properly aligned and sorted
3. **Feature Lag**: Some features use lagged values, so recent predictions may have higher uncertainty
4. **Market Conditions**: Model performance may vary across different market conditions

## 🔄 Model Updates

To retrain models with new data:

1. Add new data files to `data/` directory
2. Update the symbol list in `train_model.py` if needed
3. Run `python train_model.py`
4. Models will be automatically saved to `models/` directory

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions, please create an issue in the repository.