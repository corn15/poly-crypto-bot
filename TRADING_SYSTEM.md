# Trading System Documentation

## Overview

The position and trade tracking system provides comprehensive trade recording, position management, and PnL calculation capabilities for the cryptocurrency prediction trading system. It integrates seamlessly with the existing ML prediction pipeline to track all trading activity and maintain accurate position states.

## ✨ Features

- **📊 Position Tracking**: Real-time position management with average cost basis calculation
- **📈 PnL Calculation**: Both realized and unrealized PnL tracking with detailed breakdowns
- **💾 Trade Recording**: Complete trade history with signal metadata persistence
- **🔄 Data Recovery**: Position validation and recovery from trade history
- **📱 SQLite Storage**: Lightweight, file-based database with full ACID compliance
- **🎯 Signal Integration**: Links trades to ML prediction signals for performance analysis
- **⚡ Performance Monitoring**: Portfolio-level statistics and individual symbol performance

## 🏗 Architecture

### Core Components

```
src/
├── core/
│   ├── model/
│   │   ├── trade.py          # Trade data models
│   │   └── position.py       # Position management models
│   └── service/
│       ├── position_manager.py  # Position tracking logic
│       ├── trade_recorder.py    # Trade recording service
│       └── trading.py          # Unified trading service
└── adapter/
    └── database/
        └── sqlite_adapter.py   # Database operations
```

### Data Flow

```
ML Signal → Trading Decision → Trade Execution → Position Update → PnL Calculation
     ↓            ↓                 ↓              ↓              ↓
Signal Metadata → Trade Record → Database → Position State → Performance Metrics
```

## 📊 Database Schema

### Trades Table
```sql
CREATE TABLE trades (
    trade_id TEXT PRIMARY KEY,           -- UUID for each trade
    timestamp INTEGER NOT NULL,          -- Milliseconds timestamp
    symbol TEXT NOT NULL,               -- Trading pair (e.g., BTCUSDT)
    side TEXT NOT NULL,                 -- BUY or SELL
    price REAL NOT NULL,                -- Execution price
    size REAL NOT NULL,                 -- Trade size (always positive)
    fee REAL NOT NULL,                  -- Trading fees
    signal_metadata TEXT NOT NULL       -- JSON signal data
);
```

### Positions Table
```sql
CREATE TABLE positions (
    symbol TEXT PRIMARY KEY,            -- Trading pair
    size REAL NOT NULL,                 -- Position size (+long, -short)
    average_price REAL NOT NULL,        -- Weighted average cost basis
    last_update_ts INTEGER NOT NULL     -- Last update timestamp
);
```

## 🔧 Core Models

### Trade Model

```python
@dataclass
class Trade:
    symbol: str                    # Trading pair
    side: TradeSide               # BUY or SELL
    price: float                  # Execution price
    size: float                   # Trade size (positive)
    fee: float                    # Trading fee
    timestamp: int                # Milliseconds timestamp
    signal_metadata: Dict         # Signal information
    trade_id: str                # UUID (auto-generated)

    @property
    def total_value(self) -> float:
        """Total value excluding fees"""
        return self.price * self.size

    @property  
    def net_value(self) -> float:
        """Net cash flow including fees"""
        if self.side == TradeSide.BUY:
            return -(self.total_value + self.fee)
        else:
            return self.total_value - self.fee
```

### Position Model

```python
@dataclass
class Position:
    symbol: str                   # Trading pair
    size: float                   # Position size (+long, -short)
    average_price: float          # Weighted average cost basis  
    last_update_ts: int          # Last update timestamp

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL"""
        return self.size * (current_price - self.average_price)

    def calculate_realized_pnl(self, exit_price: float, exit_size: float) -> float:
        """Calculate realized PnL for partial/full exit"""
        exit_size_abs = abs(exit_size)
        return exit_size_abs * (exit_price - self.average_price) * (1 if self.size > 0 else -1)
```

## 🚀 Usage Examples

### Basic Trade Recording

```python
from pathlib import Path
from src.core.service.trading import TradingService
from src.core.model.trade import TradeSide
from src.core.model.signal import SignalResult, MarketSignal

# Initialize trading service
trading_service = TradingService(Path("data/trading.db"))

# Create signal result
signal = SignalResult(
    signal=MarketSignal.BULLISH,
    probability=0.75,
    reasoning="Strong technical indicators suggest upward movement"
)

# Record a trade
trade = trading_service.record_trade(
    symbol="BTCUSDT",
    side=TradeSide.BUY,
    price=42000.0,
    size=0.5,
    fee=10.5,
    timestamp=int(time.time() * 1000),
    signal_result=signal
)
```

### Position Management

```python
# Get current position
position = trading_service.get_position("BTCUSDT")
if position:
    print(f"Size: {position.size}")
    print(f"Average Price: ${position.average_price:.2f}")
    print(f"Direction: {'LONG' if position.is_long else 'SHORT'}")

# Calculate unrealized PnL
current_price = 45000.0
unrealized_pnl = trading_service.calculate_unrealized_pnl("BTCUSDT", current_price)
print(f"Unrealized PnL: ${unrealized_pnl:.2f}")
```

### Portfolio Analysis

```python
# Get portfolio statistics
current_prices = {"BTCUSDT": 45000.0, "ETHUSDT": 2800.0}
portfolio_stats = trading_service.get_portfolio_stats(current_prices)

print("Portfolio Overview:")
print(f"Total Positions: {portfolio_stats['positions']['total_positions']}")
print(f"Total Unrealized PnL: ${portfolio_stats['positions']['total_unrealized_pnl']:.2f}")
print(f"Total Realized PnL: ${portfolio_stats['trading']['total_realized_pnl']:.2f}")
print(f"Total Fees: ${portfolio_stats['trading']['total_fees']:.2f}")
```

### Trade History Analysis

```python
# Get trade history
trades = trading_service.get_trade_history("BTCUSDT")
print(f"Total trades for BTCUSDT: {len(trades)}")

# Get trading PnL breakdown
pnl_data = trading_service.calculate_trading_pnl("BTCUSDT")
print(f"Realized PnL: ${pnl_data['realized_pnl']:.4f}")
print(f"Total Fees: ${pnl_data['total_fees']:.4f}")
print(f"Net PnL: ${pnl_data['net_pnl']:.4f}")
```

## 🔗 Strategy Integration

### Connecting to ML Signals

```python
from src.core.service.strategy import StrategyService

class SignalBasedTrader:
    def __init__(self, trading_service: TradingService, strategy_service: StrategyService):
        self.trading_service = trading_service
        self.strategy_service = strategy_service

    def process_signals_and_trade(self):
        """Process ML signals and execute trades"""
        signals = self.strategy_service.generate_signals_for_all_assets()
        
        for asset, signal_result in signals.items():
            if signal_result.signal == MarketSignal.BULLISH and signal_result.probability > 0.7:
                # Execute buy trade
                current_price = self.get_current_price(asset)
                trade = self.trading_service.record_trade(
                    symbol=asset,
                    side=TradeSide.BUY,
                    price=current_price,
                    size=self.calculate_position_size(current_price),
                    fee=current_price * 0.001,  # 0.1% fee
                    timestamp=int(time.time() * 1000),
                    signal_result=signal_result
                )
                
                if trade:
                    logger.info(f"Executed trade based on signal: {trade.trade_id}")
```

### Performance Tracking by Signal Type

```python
# Export trades with signal metadata for analysis
trades = trading_service.export_trades_to_dict("BTCUSDT")

# Analyze performance by signal strength
strong_signals = [t for t in trades if t['signal_metadata'].get('probability', 0) > 0.8]
weak_signals = [t for t in trades if t['signal_metadata'].get('probability', 0) < 0.6]

print(f"Trades from strong signals: {len(strong_signals)}")
print(f"Trades from weak signals: {len(weak_signals)}")
```

## ⚙️ Configuration

### Database Configuration

Add to your `config/application.yaml`:

```yaml
database:
  db_path: "data/trading.db"  # Path to SQLite database file
```

### Application Integration

Update `main.py` to include the trading service:

```python
from src.core.service.trading import TradingService

def main():
    config = ApplicationConfig.load(Path(args.config))
    
    # Initialize trading service
    trading_service = TradingService(Path(config.database.db_path))
    
    # Pass to other services as needed
    # ...
```

## 📋 API Reference

### TradingService

#### Methods

**`record_trade(symbol, side, price, size, fee, timestamp, signal_result=None)`**
- Records a new trade and updates positions
- Returns `Trade` object or `None` if failed

**`get_position(symbol)`**  
- Returns current `Position` for symbol or `None`

**`get_all_positions()`**
- Returns dictionary of all open positions

**`calculate_unrealized_pnl(symbol, current_price)`**
- Returns unrealized PnL as float

**`calculate_total_unrealized_pnl(prices)`**
- Returns total portfolio unrealized PnL

**`get_portfolio_stats(prices=None)`**
- Returns comprehensive portfolio statistics dictionary

**`close_position(symbol, price, fee, timestamp, signal_result=None)`**
- Closes entire position at specified price
- Returns `Trade` object for the closing trade

**`validate_data_integrity()`**
- Validates all positions against trade history
- Returns `True` if all positions are valid

**`recover_from_trades(symbol=None)`**
- Recovers positions by recalculating from trade history
- Returns `True` if recovery successful

## 🧪 Testing

### Run Comprehensive Tests

```bash
# Run the demonstration script
python3 test_trading_system.py

# Run integration example
python3 trading_integration_example.py
```

### Test Coverage

The test suite covers:
- ✅ Basic trade recording
- ✅ Position updates and average cost basis calculation  
- ✅ Realized and unrealized PnL calculations
- ✅ Position closure and reversal
- ✅ Short position handling
- ✅ Data validation and recovery
- ✅ Portfolio statistics
- ✅ Trade history export
- ✅ Signal metadata persistence

### Validation Features

```python
# Validate data integrity
if trading_service.validate_data_integrity():
    print("✅ All positions validated")
else:
    print("❌ Validation failed - running recovery")
    trading_service.recover_from_trades()
```

## 📊 Performance Considerations

### Database Optimization

- Automatic indexing on `symbol` and `timestamp` columns
- Efficient queries with proper WHERE clauses
- Connection pooling with context managers

### Memory Management

- Positions loaded into memory for fast access
- Lazy loading of trade history when needed
- Automatic cleanup of flat positions

### Scalability

- SQLite handles millions of trade records efficiently
- Pagination support for large result sets
- Configurable database location for distributed setups

## 🚨 Error Handling

### Common Issues and Solutions

**Position Validation Failures**
```python
if not trading_service.validate_data_integrity():
    logger.warning("Position validation failed")
    if trading_service.recover_from_trades():
        logger.info("Recovery successful") 
    else:
        logger.error("Manual intervention required")
```

**Database Connection Issues**
- Database automatically created if missing
- Proper exception handling with detailed logging
- Transaction rollback on failures

**PnL Calculation Errors**
- Handles edge cases like zero positions
- Floating point precision considerations
- Comprehensive logging for debugging

## 🔍 Monitoring and Logging

### Key Metrics to Monitor

- Trade execution success rate
- Position validation status  
- Database query performance
- PnL calculation accuracy
- Signal-to-trade conversion rate

### Log Levels

- `INFO`: Trade executions, position updates, PnL calculations
- `WARNING`: Validation failures, recovery attempts
- `ERROR`: Database errors, calculation failures
- `DEBUG`: Detailed trade flow information

## 🎯 Best Practices

1. **Always validate positions after system restarts**
2. **Use signal metadata to track strategy performance**
3. **Monitor total fees vs realized PnL**
4. **Backup database regularly for production use**
5. **Test with small positions before scaling up**
6. **Use proper error handling in production code**

## 📈 Future Enhancements

- **Multi-currency support**: Handle different base currencies
- **Advanced order types**: Stop losses, take profits
- **Risk management**: Position sizing, exposure limits
- **Performance analytics**: Sharpe ratio, maximum drawdown
- **Real-time WebSocket**: Live position updates
- **API endpoints**: REST API for external integration

---

*For questions or issues, check the test files for working examples or refer to the inline code documentation.*