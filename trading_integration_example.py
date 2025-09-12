#!/usr/bin/env python3
"""
Integration example showing how to connect the trading system with strategy signals.

This example demonstrates:
1. Processing strategy signals and converting them to trades
2. Recording trades with signal metadata
3. Position management based on ML predictions
4. PnL tracking from signal-driven trading
"""

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from src.core.model.signal import MarketSignal, SignalResult
from src.core.model.trade import TradeSide
from src.core.service.trading import TradingService
from src.prediction.predictor import CryptoPredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class SignalBasedTrader:
    """
    Example trader that converts ML prediction signals into actual trades
    and manages positions based on signal strength and market conditions.
    """

    def __init__(self, trading_service: TradingService, predictor: CryptoPredictor):
        self.trading_service = trading_service
        self.predictor = predictor

        # Trading parameters
        self.max_position_size = 1000.0  # Maximum position size in USD
        self.min_signal_strength = 0.6  # Minimum probability to trade
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.fee_rate = 0.001  # 0.1% fee rate

    def process_signal(self, symbol: str, candle_data: pl.DataFrame, current_price: float) -> bool:
        """
        Process ML prediction signal and potentially execute trades

        Returns True if a trade was executed, False otherwise
        """
        try:
            # Get prediction from ML model
            probability = self.predictor.predict(candle_data)

            # Convert probability to signal
            signal_result = self._probability_to_signal(probability)

            logger.info(
                f"Signal for {symbol}: {signal_result.signal.value if signal_result.signal else 'NEUTRAL'} "
                f"(probability: {probability:.4f})"
            )

            # Get current position
            current_position = self.trading_service.get_position(symbol)

            # Determine if we should trade
            trade_decision = self._make_trade_decision(
                symbol, signal_result, current_position, current_price
            )

            if trade_decision:
                return self._execute_trade(symbol, trade_decision, current_price, signal_result)

            return False

        except Exception as e:
            logger.error(f"Error processing signal for {symbol}: {e}")
            return False

    def _probability_to_signal(self, probability: float) -> SignalResult:
        """Convert ML probability to trading signal"""
        if probability >= 0.7:
            signal = MarketSignal.BULLISH
            reasoning = f"High bullish probability ({probability:.3f})"
        elif probability >= self.min_signal_strength:
            signal = MarketSignal.BULLISH
            reasoning = f"Moderate bullish probability ({probability:.3f})"
        elif probability <= 0.3:
            signal = MarketSignal.BEARISH
            reasoning = f"High bearish probability ({probability:.3f})"
        elif probability <= 0.4:
            signal = MarketSignal.BEARISH
            reasoning = f"Moderate bearish probability ({probability:.3f})"
        else:
            signal = MarketSignal.NEUTRAL
            reasoning = f"Neutral probability ({probability:.3f})"

        return SignalResult(signal=signal, probability=probability, reasoning=reasoning, error=None)

    def _make_trade_decision(
        self,
        symbol: str,
        signal_result: SignalResult,
        current_position: Optional[object],
        current_price: float,
    ) -> Optional[Dict]:
        """
        Decide whether to trade based on signal and current position

        Returns trade decision dict or None if no trade should be made
        """
        if not signal_result.signal or signal_result.signal == MarketSignal.NEUTRAL:
            # Check if we should close existing position
            if current_position and not current_position.is_flat:
                # Close position on neutral signal if we have significant unrealized loss
                unrealized_pnl = self.trading_service.calculate_unrealized_pnl(
                    symbol, current_price
                )
                loss_threshold = current_position.total_cost * -0.05  # 5% stop loss

                if unrealized_pnl < loss_threshold:
                    return {
                        "action": "close",
                        "reason": f"Stop loss triggered: ${unrealized_pnl:.2f}",
                    }
            return None

        # Don't trade if signal strength is too weak
        if abs(signal_result.probability - 0.5) < (self.min_signal_strength - 0.5):
            return None

        # Determine trade action based on signal and position
        if signal_result.signal == MarketSignal.BULLISH:
            if not current_position or current_position.is_flat:
                # Open long position
                return {"action": "buy", "reason": "Opening long position on bullish signal"}
            elif current_position.is_short:
                # Close short and go long
                return {"action": "reverse_to_long", "reason": "Reversing short to long position"}
            else:
                # Already long - could add to position if signal is very strong
                if signal_result.probability > 0.8:
                    return {
                        "action": "add_long",
                        "reason": "Adding to long position on very strong signal",
                    }

        elif signal_result.signal == MarketSignal.BEARISH:
            if not current_position or current_position.is_flat:
                # Open short position
                return {"action": "sell", "reason": "Opening short position on bearish signal"}
            elif current_position.is_long:
                # Close long position
                return {"action": "close_long", "reason": "Closing long position on bearish signal"}

        return None

    def _execute_trade(
        self, symbol: str, trade_decision: Dict, current_price: float, signal_result: SignalResult
    ) -> bool:
        """Execute the decided trade"""
        try:
            action = trade_decision["action"]
            current_position = self.trading_service.get_position(symbol)
            timestamp = int(time.time() * 1000)

            if action == "buy":
                # Calculate position size based on risk management
                trade_size = self._calculate_position_size(current_price, "buy")
                fee = trade_size * current_price * self.fee_rate

                trade = self.trading_service.record_trade(
                    symbol=symbol,
                    side=TradeSide.BUY,
                    price=current_price,
                    size=trade_size,
                    fee=fee,
                    timestamp=timestamp,
                    signal_result=signal_result,
                )

                if trade:
                    logger.info(
                        f"✅ Opened long position: {trade_size:.4f} {symbol} @ ${current_price:.2f}"
                    )
                    return True

            elif action == "sell":
                # Open short position
                trade_size = self._calculate_position_size(current_price, "sell")
                fee = trade_size * current_price * self.fee_rate

                trade = self.trading_service.record_trade(
                    symbol=symbol,
                    side=TradeSide.SELL,
                    price=current_price,
                    size=trade_size,
                    fee=fee,
                    timestamp=timestamp,
                    signal_result=signal_result,
                )

                if trade:
                    logger.info(
                        f"✅ Opened short position: {trade_size:.4f} {symbol} @ ${current_price:.2f}"
                    )
                    return True

            elif action == "close":
                # Close entire position
                trade = self.trading_service.close_position(
                    symbol=symbol,
                    price=current_price,
                    fee=abs(current_position.size) * current_price * self.fee_rate,
                    timestamp=timestamp,
                    signal_result=signal_result,
                )

                if trade:
                    logger.info(
                        f"✅ Closed position: {abs(current_position.size):.4f} {symbol} @ ${current_price:.2f}"
                    )
                    return True

            elif action == "close_long":
                # Close long position
                if current_position and current_position.is_long:
                    fee = current_position.size * current_price * self.fee_rate
                    trade = self.trading_service.record_trade(
                        symbol=symbol,
                        side=TradeSide.SELL,
                        price=current_price,
                        size=current_position.size,
                        fee=fee,
                        timestamp=timestamp,
                        signal_result=signal_result,
                    )

                    if trade:
                        logger.info(
                            f"✅ Closed long position: {current_position.size:.4f} {symbol} @ ${current_price:.2f}"
                        )
                        return True

            elif action == "add_long":
                # Add to existing long position (if not too large)
                if current_position and current_position.total_cost < self.max_position_size * 0.7:
                    additional_size = self._calculate_position_size(current_price, "buy") * 0.5
                    fee = additional_size * current_price * self.fee_rate

                    trade = self.trading_service.record_trade(
                        symbol=symbol,
                        side=TradeSide.BUY,
                        price=current_price,
                        size=additional_size,
                        fee=fee,
                        timestamp=timestamp,
                        signal_result=signal_result,
                    )

                    if trade:
                        logger.info(
                            f"✅ Added to long position: {additional_size:.4f} {symbol} @ ${current_price:.2f}"
                        )
                        return True

            return False

        except Exception as e:
            logger.error(f"Failed to execute trade for {symbol}: {e}")
            return False

    def _calculate_position_size(self, price: float, side: str) -> float:
        """Calculate position size based on risk management rules"""
        # Risk-based position sizing
        risk_amount = self.max_position_size * self.risk_per_trade

        # Calculate size based on price and risk
        # For crypto, we'll use a simple fixed risk amount
        size = risk_amount / price

        # Ensure minimum viable size (e.g., $10 worth)
        min_value = 10.0
        min_size = min_value / price

        return max(size, min_size)

    def get_performance_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get comprehensive performance summary"""
        try:
            portfolio_stats = self.trading_service.get_portfolio_stats(current_prices)

            # Get all symbols we've traded
            all_positions = self.trading_service.get_all_positions()
            traded_symbols = set(all_positions.keys())

            # Add symbols that may have been closed
            db_stats = self.trading_service.get_database_stats()

            summary = {
                "portfolio_stats": portfolio_stats,
                "trading_performance": {},
                "current_positions": len(all_positions),
                "total_trades": db_stats.get("total_trades", 0),
            }

            # Get performance for each symbol
            for symbol in traded_symbols:
                if symbol in current_prices:
                    perf = self.trading_service.get_symbol_performance(
                        symbol, current_prices[symbol]
                    )
                    summary["trading_performance"][symbol] = perf

            return summary

        except Exception as e:
            logger.error(f"Failed to get performance summary: {e}")
            return {}


def simulate_trading_session():
    """
    Simulate a trading session with the integrated trading system
    """
    print("=" * 60)
    print(" SIGNAL-BASED TRADING INTEGRATION EXAMPLE")
    print("=" * 60)

    # Initialize services
    db_path = Path("data/signal_trading_example.db")
    if db_path.exists():
        db_path.unlink()

    trading_service = TradingService(db_path)
    predictor = CryptoPredictor("models")

    # Create signal-based trader
    trader = SignalBasedTrader(trading_service, predictor)

    # Simulate market data (normally this would come from live feeds)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    # Mock current prices
    current_prices = {"BTCUSDT": 43500.0, "ETHUSDT": 2750.0, "SOLUSDT": 65.0}

    print(f"\n📊 Processing signals for {len(symbols)} symbols...")

    # Simulate processing signals for each symbol
    trades_executed = 0

    for symbol in symbols:
        print(f"\n🔍 Analyzing {symbol}...")

        # In a real system, this would be live 5-minute candle data
        # For demo, we'll create mock data that would trigger our predictor
        try:
            # Load sample data (you would replace this with live data)
            sample_data_path = Path(f"data/{symbol}_5m_20240901.parquet")
            if sample_data_path.exists():
                df = pl.read_parquet(sample_data_path)

                # Process signal for the latest data point
                if trader.process_signal(symbol, df, current_prices[symbol]):
                    trades_executed += 1

                    # Show position after trade
                    position = trading_service.get_position(symbol)
                    if position:
                        unrealized_pnl = trading_service.calculate_unrealized_pnl(
                            symbol, current_prices[symbol]
                        )
                        print(f"   Position: {position.size:.4f} @ ${position.average_price:.2f}")
                        print(f"   Unrealized PnL: ${unrealized_pnl:.4f}")

            else:
                print(f"   ⚠️  No sample data available for {symbol}")

        except Exception as e:
            print(f"   ❌ Error processing {symbol}: {e}")

    print(f"\n📈 Executed {trades_executed} trades based on ML signals")

    # Show final performance summary
    print("\n" + "=" * 40)
    print(" FINAL PERFORMANCE SUMMARY")
    print("=" * 40)

    performance = trader.get_performance_summary(current_prices)

    # Portfolio overview
    portfolio = performance.get("portfolio_stats", {})
    positions_stats = portfolio.get("positions", {})
    trading_stats = portfolio.get("trading", {})
    combined_stats = portfolio.get("combined", {})

    print("\nPortfolio Overview:")
    print(f"  Open Positions: {performance.get('current_positions', 0)}")
    print(f"  Total Trades: {performance.get('total_trades', 0)}")
    print(f"  Long Positions: {positions_stats.get('long_positions', 0)}")
    print(f"  Short Positions: {positions_stats.get('short_positions', 0)}")

    if combined_stats:
        print("\nPnL Summary:")
        print(f"  Total PnL: ${combined_stats.get('total_pnl', 0):.4f}")
        print(f"  Net PnL (after fees): ${combined_stats.get('net_pnl', 0):.4f}")
        print(f"  Total Fees: ${combined_stats.get('total_fees', 0):.4f}")

    # Individual symbol performance
    trading_perf = performance.get("trading_performance", {})
    if trading_perf:
        print("\nSymbol Performance:")
        for symbol, perf in trading_perf.items():
            print(f"  {symbol}:")
            pos = perf.get("position", {})
            if pos.get("size", 0) != 0:
                print(f"    Position: {pos['size']:.4f} ({pos['direction']})")
                print(f"    Avg Price: ${pos['average_price']:.2f}")
                if "unrealized_pnl" in perf:
                    print(f"    Unrealized PnL: ${perf['unrealized_pnl']:.4f}")

            trading_pnl = perf.get("trading_pnl", {})
            if trading_pnl.get("trade_count", 0) > 0:
                print(f"    Realized PnL: ${trading_pnl['realized_pnl']:.4f}")
                print(f"    Trade Count: {trading_pnl['trade_count']}")

    print("\n✅ Integration example completed!")
    print(f"📁 Database saved at: {db_path}")

    # Export trades for analysis
    trades_export = trading_service.export_trades_to_dict()
    export_file = Path("data/signal_trades_export.json")

    import json

    export_file.parent.mkdir(exist_ok=True)
    with open(export_file, "w") as f:
        json.dump(trades_export, f, indent=2, default=str)

    print(f"📁 Trade history exported to: {export_file}")


if __name__ == "__main__":
    simulate_trading_session()
