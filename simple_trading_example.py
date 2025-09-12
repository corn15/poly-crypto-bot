#!/usr/bin/env python3
"""
Simple example demonstrating basic trading system usage.

This shows the core functionality in a minimal, easy-to-understand way.
"""

import time
from pathlib import Path

from src.core.model.signal import MarketSignal, SignalResult
from src.core.model.trade import TradeSide
from src.core.service.trading import TradingService


def main():
    print("🔄 Simple Trading System Example")
    print("=" * 40)

    # Initialize trading service with a test database
    db_path = Path("data/simple_example.db")
    if db_path.exists():
        db_path.unlink()
        print("Cleaned up previous database")

    trading_service = TradingService(db_path)
    print(f"✅ Initialized trading service: {db_path}")

    # Current timestamp
    now = int(time.time() * 1000)

    # Example 1: Record a buy trade
    print("\n📈 Recording BTC buy trade...")

    signal = SignalResult(
        signal=MarketSignal.BULLISH,
        probability=0.72,
        reasoning="Technical analysis shows bullish momentum",
    )

    trade1 = trading_service.record_trade(
        symbol="BTCUSDT",
        side=TradeSide.BUY,
        price=42500.0,
        size=0.1,
        fee=2.125,
        timestamp=now,
        signal_result=signal,
    )

    if trade1:
        print(f"✅ Trade recorded: {trade1.trade_id[:8]}")
        print(f"   Buy: {trade1.size} BTC @ ${trade1.price:,.2f}")
        print(f"   Fee: ${trade1.fee:.2f}")

    # Check position
    position = trading_service.get_position("BTCUSDT")
    if position:
        print("\n📊 Current BTC Position:")
        print(f"   Size: {position.size} BTC")
        print(f"   Average Price: ${position.average_price:,.2f}")
        print(f"   Total Cost: ${position.total_cost:,.2f}")

    # Example 2: Calculate unrealized PnL
    current_btc_price = 44000.0
    unrealized_pnl = trading_service.calculate_unrealized_pnl("BTCUSDT", current_btc_price)

    print("\n💰 Unrealized PnL:")
    print(f"   Current Price: ${current_btc_price:,.2f}")
    print(f"   Unrealized PnL: ${unrealized_pnl:.2f}")
    print(f"   Return: {(unrealized_pnl / position.total_cost) * 100:.2f}%")

    # Example 3: Partial sell
    print("\n📉 Recording partial BTC sell...")

    sell_signal = SignalResult(
        signal=MarketSignal.NEUTRAL,
        probability=0.48,
        reasoning="Taking partial profits at resistance",
    )

    trade2 = trading_service.record_trade(
        symbol="BTCUSDT",
        side=TradeSide.SELL,
        price=43800.0,
        size=0.05,
        fee=1.095,
        timestamp=now + 3600000,  # 1 hour later
        signal_result=sell_signal,
    )

    if trade2:
        print(f"✅ Partial sell: {trade2.size} BTC @ ${trade2.price:,.2f}")

    # Check updated position
    position = trading_service.get_position("BTCUSDT")
    if position:
        print("\n📊 Updated BTC Position:")
        print(f"   Size: {position.size} BTC")
        print(f"   Average Price: ${position.average_price:,.2f}")

    # Example 4: Trading PnL summary
    pnl_summary = trading_service.calculate_trading_pnl("BTCUSDT")
    print("\n💵 Trading PnL Summary:")
    print(f"   Realized PnL: ${pnl_summary['realized_pnl']:.2f}")
    print(f"   Total Fees: ${pnl_summary['total_fees']:.2f}")
    print(f"   Net PnL: ${pnl_summary['net_pnl']:.2f}")
    print(f"   Trade Count: {pnl_summary['trade_count']}")

    # Example 5: Portfolio stats
    current_prices = {"BTCUSDT": current_btc_price}
    portfolio = trading_service.get_portfolio_stats(current_prices)

    print("\n🎯 Portfolio Summary:")
    pos_stats = portfolio.get("positions", {})
    print(f"   Total Positions: {pos_stats.get('total_positions', 0)}")
    print(f"   Total Cost Basis: ${pos_stats.get('total_cost_basis', 0):,.2f}")
    print(f"   Total Unrealized PnL: ${pos_stats.get('total_unrealized_pnl', 0):.2f}")

    combined = portfolio.get("combined", {})
    if combined:
        print(f"   Net Total PnL: ${combined.get('net_pnl', 0):.2f}")

    # Example 6: Recent trades
    recent_trades = trading_service.get_recent_trades("BTCUSDT", 5)
    print(f"\n📋 Recent Trades ({len(recent_trades)}):")
    for trade in recent_trades:
        side_emoji = "🟢" if trade.side == TradeSide.BUY else "🔴"
        print(
            f"   {side_emoji} {trade.side.value} {trade.size} @ ${trade.price:,.2f} (fee: ${trade.fee:.2f})"
        )

    print("\n✅ Example completed successfully!")
    print(f"📁 Database saved at: {db_path}")

    # Quick data validation
    if trading_service.validate_data_integrity():
        print("✅ Data integrity validated")
    else:
        print("⚠️ Data integrity issues detected")


if __name__ == "__main__":
    main()
