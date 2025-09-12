# StrategyService Documentation

## Overview

The StrategyService generates automated trading signals by combining ML predictions with real-time market data from Binance and Polymarket.

## How It Works

1. **Data Collection**: Fetches 3 hours of 5-minute candles from Binance
2. **Prediction**: Uses ML models to predict probability of price increase in next hour
3. **Market Data**: Gets current order book prices from Polymarket
4. **Signal Generation**: Applies trading rules to generate BUY/SELL/WAIT signals
5. **Execution**: Runs automatically every 5 minutes for all configured assets

## Trading Rules (Priority Order)

The service applies these rules in sequence:

1. `probability > 0.65 AND probability > token1_ask` → **BUY**
2. `probability < 0.4 AND probability < token1_bid` → **SELL**
3. `probability < 0.4 AND (1-probability) > token2_ask` → **BUY**
4. `probability > 0.65 AND (1-probability) > token2_bid` → **SELL**
5. `0.4 ≤ probability ≤ 0.65` → **WAIT**

Where:
- `token1` = UP token (betting price will go up)
- `token2` = DOWN token (betting price will go down)
- `probability` = ML prediction of price increase (0-1)

## Asset Mapping

| Polymarket | Binance |
|------------|---------|
| bitcoin    | BTCUSDT |
| ethereum   | ETHUSDT |
| solana     | SOLUSDT |

## Usage

### Automatic Mode (Recommended)
```bash
python main.py --config config/config.yaml
```
Runs continuously, generating signals every 5 minutes.

### Programmatic Usage
```python
from src.application.strategy_service import StrategyService

# Initialize
strategy_service = StrategyService(binance, polymarket, predictor)

# Single asset
result = strategy_service.generate_signal_for_asset("bitcoin")
print(f"Signal: {result.signal}")
print(f"Probability: {result.probability}")
print(f"Reasoning: {result.reasoning}")

# All assets
results = strategy_service.generate_signals_for_all_assets()
```

### Testing
```bash
python test_strategy_service.py
```

## SignalResult Structure

```python
@dataclass
class SignalResult:
    signal: Optional[Signal]      # BUY, SELL, WAIT, or None if error
    probability: Optional[float]  # ML prediction (0-1)
    reasoning: str               # Human-readable explanation
    error: Optional[str]         # Error message if failed
```

## Configuration

Required in `config/config.yaml`:

```yaml
polymarket:
  assets: ["bitcoin", "ethereum", "solana"]

binance:
  candle_interval: "5m"
  assets: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
```

## Error Handling

- Returns `SignalResult` with `signal=None` and error details
- Never raises exceptions
- Continues processing other assets if one fails
- Logs all errors for debugging

## Caching

- Binance candle data cached for 60 seconds
- Polymarket prices fetched fresh (change frequently)
- ML models loaded once at startup

## Performance

- Execution time: ~5-10 seconds per cycle
- Memory usage: Minimal
- API calls: ~4-6 per asset (reduced by caching)

## Example Output

```
bitcoin: BUY - High prob (0.724) > token1 ask (0.680)
ethereum: WAIT - Neutral prob (0.523) in [0.4, 0.65]
solana: SELL - Low prob (0.342) < token1 bid (0.385)
```

## Files

- `src/application/strategy_service.py` - Main implementation
- `src/core/model/signal.py` - Signal and result structures
- `test_strategy_service.py` - Basic test suite
- `config/config.yaml` - Configuration