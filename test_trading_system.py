#!/usr/bin/env python3
"""
Comprehensive test demonstrating the position and trade tracking system.
This test shows all major features including trade recording, position management,
PnL calculations, and data persistence.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict

from src.core.model.signal import MarketSignal, SignalResult
from src.core.model.trade import TradeSide
from src.core.service.trading import TradingService

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def create_test_signal_result(
    signal: MarketSignal, probability: float, reasoning: str
) -> SignalResult:
    """Create a test signal result"""
    return SignalResult(signal=signal, probability=probability, reasoning=reasoning, error=None)


def get_current_timestamp() -> int:
    """Get current timestamp in milliseconds"""
    return int(time.time() * 1000)


def print_section(title: str):
    """Print a section header"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_position_summary(trading_service: TradingService, prices: Dict[str, float] = None):
    """Print detailed position summary"""
    positions = trading_service.get_all_positions()

    if not positions:
        print("No open positions")
        return

    print(f"\nOpen Positions ({len(positions)}):")
    print("-" * 80)
    print(f"{'Symbol':<10} {'Size':<12} {'Avg Price':<12} {'Direction':<8} {'Unrealized PnL':<15}")
    print("-" * 80)

    for symbol, position in positions.items():
        direction = "LONG" if position.is_long else "SHORT"
        unrealized_pnl = ""

        if prices and symbol in prices:
            pnl = trading_service.calculate_unrealized_pnl(symbol, prices[symbol])
            unrealized_pnl = f"${pnl:.4f}"

        print(
            f"{symbol:<10} {position.size:<12.4f} ${position.average_price:<11.4f} {direction:<8} {unrealized_pnl:<15}"
        )


def print_trading_pnl(trading_service: TradingService, symbol: str):
    """Print trading PnL for a symbol"""
    pnl_data = trading_service.calculate_trading_pnl(symbol)
    print(f"\nTrading PnL for {symbol}:")
    print(f"  Realized PnL: ${pnl_data['realized_pnl']:.4f}")
    print(f"  Total Fees: ${pnl_data['total_fees']:.4f}")
    print(f"  Net PnL: ${pnl_data['net_pnl']:.4f}")
    print(f"  Trade Count: {pnl_data['trade_count']}")


def main():
    """Main test function demonstrating the trading system"""

    print_section("TRADING SYSTEM DEMONSTRATION")

    # Initialize trading service with test database
    test_db_path = Path("data/test_trading.db")

    # Remove existing test database
    if test_db_path.exists():
        test_db_path.unlink()
        print(f"Removed existing test database: {test_db_path}")

    # Create trading service
    trading_service = TradingService(test_db_path)
    print(f"Initialized trading service with database: {test_db_path}")

    # Test symbols and prices
    btc_symbol = "BTCUSDT"
    eth_symbol = "ETHUSDT"

    # Current market prices (simulated)
    market_prices = {btc_symbol: 45000.0, eth_symbol: 2800.0}

    print_section("1. RECORDING INITIAL TRADES")

    # Record some initial BTC trades
    print("\n📈 Recording BTC trades...")

    # BTC Buy #1
    btc_buy_signal = create_test_signal_result(
        MarketSignal.BULLISH, 0.72, "High probability bullish signal based on technical indicators"
    )

    trade1 = trading_service.record_trade(
        symbol=btc_symbol,
        side=TradeSide.BUY,
        price=42000.0,
        size=0.5,
        fee=10.5,
        timestamp=get_current_timestamp(),
        signal_result=btc_buy_signal,
    )
    print(f"✅ Recorded BTC buy: {trade1.trade_id[:8]}")

    # BTC Buy #2 (adding to position)
    trade2 = trading_service.record_trade(
        symbol=btc_symbol,
        side=TradeSide.BUY,
        price=43000.0,
        size=0.3,
        fee=6.45,
        timestamp=get_current_timestamp(),
        signal_result=btc_buy_signal,
    )
    print(f"✅ Recorded BTC buy: {trade2.trade_id[:8]}")

    print_position_summary(trading_service, market_prices)

    print_section("2. RECORDING ETH TRADES")

    # ETH trades
    eth_signal = create_test_signal_result(
        MarketSignal.BULLISH, 0.68, "Ethereum showing strong momentum with volume confirmation"
    )

    # ETH Buy
    trade3 = trading_service.record_trade(
        symbol=eth_symbol,
        side=TradeSide.BUY,
        price=2750.0,
        size=2.0,
        fee=11.0,
        timestamp=get_current_timestamp(),
        signal_result=eth_signal,
    )
    print(f"✅ Recorded ETH buy: {trade3.trade_id[:8]}")

    print_position_summary(trading_service, market_prices)

    print_section("3. PARTIAL POSITION EXIT")

    # Partial BTC sell
    btc_sell_signal = create_test_signal_result(
        MarketSignal.NEUTRAL, 0.45, "Taking partial profits at resistance level"
    )

    trade4 = trading_service.record_trade(
        symbol=btc_symbol,
        side=TradeSide.SELL,
        price=44500.0,
        size=0.2,
        fee=4.45,
        timestamp=get_current_timestamp(),
        signal_result=btc_sell_signal,
    )
    print(f"✅ Recorded BTC partial sell: {trade4.trade_id[:8]}")

    print_position_summary(trading_service, market_prices)
    print_trading_pnl(trading_service, btc_symbol)

    print_section("4. PORTFOLIO STATISTICS")

    portfolio_stats = trading_service.get_portfolio_stats(market_prices)
    print("\nPortfolio Overview:")
    print("-" * 40)

    # Position stats
    pos_stats = portfolio_stats.get("positions", {})
    print(f"Total Positions: {pos_stats.get('total_positions', 0)}")
    print(f"Long Positions: {pos_stats.get('long_positions', 0)}")
    print(f"Short Positions: {pos_stats.get('short_positions', 0)}")
    print(f"Total Cost Basis: ${pos_stats.get('total_cost_basis', 0):.2f}")

    if "total_unrealized_pnl" in pos_stats:
        print(f"Total Unrealized PnL: ${pos_stats['total_unrealized_pnl']:.2f}")
        print(f"Portfolio Return: {pos_stats.get('portfolio_return_percent', 0):.2f}%")

    # Trading stats
    trading_stats = portfolio_stats.get("trading", {})
    db_stats = trading_stats.get("database_stats", {})
    print("\nTrading Activity:")
    print(f"Total Trades: {db_stats.get('total_trades', 0)}")
    print(f"Unique Symbols: {db_stats.get('unique_symbols', 0)}")
    print(f"Total Realized PnL: ${trading_stats.get('total_realized_pnl', 0):.4f}")
    print(f"Total Fees Paid: ${trading_stats.get('total_fees', 0):.4f}")
    print(f"Net Trading PnL: ${trading_stats.get('net_trading_pnl', 0):.4f}")

    print_section("5. INDIVIDUAL SYMBOL PERFORMANCE")

    # BTC performance
    btc_perf = trading_service.get_symbol_performance(btc_symbol, market_prices[btc_symbol])
    print("\n🟡 BTC Performance:")
    print(
        f"  Current Position: {btc_perf['position']['size']:.4f} BTC ({btc_perf['position']['direction']})"
    )
    print(f"  Average Price: ${btc_perf['position']['average_price']:.2f}")
    print(f"  Current Price: ${btc_perf.get('current_price', 0):.2f}")
    print(f"  Unrealized PnL: ${btc_perf.get('unrealized_pnl', 0):.4f}")
    print(f"  Unrealized PnL%: {btc_perf.get('unrealized_pnl_percent', 0):.2f}%")
    print(f"  Realized PnL: ${btc_perf['trading_pnl']['realized_pnl']:.4f}")
    print(f"  Total Fees: ${btc_perf['trading_pnl']['total_fees']:.4f}")

    # ETH performance
    eth_perf = trading_service.get_symbol_performance(eth_symbol, market_prices[eth_symbol])
    print("\n🔵 ETH Performance:")
    print(
        f"  Current Position: {eth_perf['position']['size']:.4f} ETH ({eth_perf['position']['direction']})"
    )
    print(f"  Average Price: ${eth_perf['position']['average_price']:.2f}")
    print(f"  Current Price: ${eth_perf.get('current_price', 0):.2f}")
    print(f"  Unrealized PnL: ${eth_perf.get('unrealized_pnl', 0):.4f}")
    print(f"  Unrealized PnL%: {eth_perf.get('unrealized_pnl_percent', 0):.2f}%")
    print(f"  Realized PnL: ${eth_perf['trading_pnl']['realized_pnl']:.4f}")

    print_section("6. POSITION CLOSURE")

    # Close entire ETH position
    print("🔴 Closing entire ETH position...")

    eth_close_signal = create_test_signal_result(
        MarketSignal.BEARISH, 0.25, "Market reversal signal - closing position for risk management"
    )

    close_trade = trading_service.close_position(
        symbol=eth_symbol,
        price=2820.0,
        fee=11.28,
        timestamp=get_current_timestamp(),
        signal_result=eth_close_signal,
    )

    if close_trade:
        print(f"✅ Position closed: {close_trade.trade_id[:8]}")
        print_trading_pnl(trading_service, eth_symbol)

    print_position_summary(trading_service, market_prices)

    print_section("7. SHORT POSITION EXAMPLE")

    # Create short position in ETH
    short_signal = create_test_signal_result(
        MarketSignal.BEARISH, 0.78, "Strong bearish divergence with high volume"
    )

    short_trade = trading_service.record_trade(
        symbol=eth_symbol,
        side=TradeSide.SELL,
        price=2800.0,
        size=1.5,
        fee=8.4,
        timestamp=get_current_timestamp(),
        signal_result=short_signal,
    )
    print(f"✅ Created ETH short position: {short_trade.trade_id[:8]}")

    print_position_summary(trading_service, market_prices)

    print_section("8. DATA VALIDATION & RECOVERY")

    print("🔍 Validating data integrity...")
    if trading_service.validate_data_integrity():
        print("✅ All positions validated successfully")
    else:
        print("❌ Data integrity issues found")
        print("🔧 Attempting recovery...")
        if trading_service.recover_from_trades():
            print("✅ Recovery successful")
        else:
            print("❌ Recovery failed")

    print_section("9. TRADE HISTORY EXPORT")

    # Export all trades
    all_trades = trading_service.export_trades_to_dict()
    print(f"\n📊 Exported {len(all_trades)} trades")

    # Show recent trades summary
    print("\nRecent Trades Summary:")
    print("-" * 90)
    print(f"{'Symbol':<10} {'Side':<5} {'Size':<12} {'Price':<12} {'Fee':<10} {'PnL Impact':<12}")
    print("-" * 90)

    for trade_dict in all_trades[-5:]:  # Last 5 trades
        symbol = trade_dict["symbol"]
        side = trade_dict["side"]
        size = trade_dict["size"]
        price = trade_dict["price"]
        fee = trade_dict["fee"]

        # Simple PnL impact indication
        pnl_impact = "Opening" if side == "BUY" else "Closing/Short"

        print(f"{symbol:<10} {side:<5} {size:<12.4f} ${price:<11.2f} ${fee:<9.2f} {pnl_impact:<12}")

    print_section("10. FINAL SUMMARY")

    final_stats = trading_service.get_database_stats()
    final_portfolio = trading_service.get_portfolio_stats(market_prices)

    print("\n🎯 Final System State:")
    print("-" * 30)
    print(f"Database: {test_db_path}")
    print(f"Total Trades: {final_stats.get('total_trades', 0)}")
    print(f"Open Positions: {final_stats.get('open_positions', 0)}")
    print(f"Symbols Traded: {final_stats.get('unique_symbols', 0)}")

    # Combined PnL
    combined = final_portfolio.get("combined", {})
    if combined:
        print(f"Total PnL: ${combined.get('total_pnl', 0):.4f}")
        print(f"Net PnL (after fees): ${combined.get('net_pnl', 0):.4f}")
        print(f"Total Fees: ${combined.get('total_fees', 0):.4f}")

    print("\n✅ Trading system demonstration completed successfully!")
    print(f"📁 Test database saved at: {test_db_path}")

    # Save trade export for analysis
    export_file = Path("data/test_trades_export.json")
    export_file.parent.mkdir(exist_ok=True)

    with open(export_file, "w") as f:
        json.dump(all_trades, f, indent=2, default=str)

    print(f"📁 Trade history exported to: {export_file}")


if __name__ == "__main__":
    main()
