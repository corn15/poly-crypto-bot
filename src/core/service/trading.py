import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.adapter.database import SQLiteAdapter
from src.core.model.position import Position
from src.core.model.signal import SignalResult
from src.core.model.trade import Trade, TradeSide
from src.core.service.position_manager import PositionManager
from src.core.service.trade_recorder import TradeRecorder

logger = logging.getLogger(__name__)


class TradingService:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._ensure_data_directory()

        # Initialize database adapter
        self.db = SQLiteAdapter(db_path)

        # Initialize managers
        self.position_manager = PositionManager(self.db)
        self.trade_recorder = TradeRecorder(self.db, self.position_manager)

        # Validate data integrity on startup
        self._startup_validation()

    def _ensure_data_directory(self) -> None:
        """Ensure the data directory exists"""
        data_dir = self.db_path.parent
        data_dir.mkdir(parents=True, exist_ok=True)

    def _startup_validation(self) -> None:
        """Validate data integrity on service startup"""
        try:
            logger.info("Validating trading data integrity...")

            # Get summary stats
            stats = self.db.get_summary_stats()
            logger.info(
                f"Database loaded: {stats.get('total_trades', 0)} trades, "
                f"{stats.get('open_positions', 0)} open positions"
            )

            # Validate positions against trade history
            if not self.position_manager.validate_all_positions():
                logger.warning("Position validation failed - attempting recovery")
                if self.position_manager.recover_positions_from_trades():
                    logger.info("Successfully recovered positions from trade history")
                else:
                    logger.error(
                        "Failed to recover positions - manual intervention may be required"
                    )
            else:
                logger.info("Position validation successful")

        except Exception as e:
            logger.error(f"Startup validation failed: {e}")

    def record_trade(
        self,
        symbol: str,
        side: TradeSide,
        price: float,
        size: float,
        fee: float,
        timestamp: int,
        signal_result: Optional[SignalResult] = None,
    ) -> Optional[Trade]:
        """Record a new trade and update positions"""
        return self.trade_recorder.record_trade(
            symbol, side, price, size, fee, timestamp, signal_result
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for a symbol"""
        return self.position_manager.get_position(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self.position_manager.get_all_positions()

    def calculate_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized PnL for a specific symbol"""
        return self.position_manager.calculate_unrealized_pnl(symbol, current_price)

    def calculate_total_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate total unrealized PnL across all positions"""
        return self.position_manager.calculate_total_unrealized_pnl(prices)

    def get_position_summary(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        """Get detailed summary of all positions with optional PnL"""
        return self.position_manager.get_position_summary(prices)

    def get_trade_history(
        self, symbol: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None
    ) -> List[Trade]:
        """Get trade history for a symbol"""
        return self.trade_recorder.get_trade_history(symbol, start_ts, end_ts)

    def get_recent_trades(self, symbol: str, limit: int = 10) -> List[Trade]:
        """Get recent trades for a symbol"""
        return self.trade_recorder.get_recent_trades(symbol, limit)

    def calculate_trading_pnl(self, symbol: str) -> Dict[str, float]:
        """Calculate realized PnL and fees for a symbol"""
        return self.trade_recorder.calculate_trading_pnl(symbol)

    def get_portfolio_stats(self, prices: Optional[Dict[str, float]] = None) -> Dict:
        """Get comprehensive portfolio statistics"""
        position_stats = self.position_manager.get_portfolio_stats(prices)
        trading_summary = self.trade_recorder.get_trading_summary()

        combined_stats = {
            "positions": position_stats,
            "trading": trading_summary,
        }

        # Add combined metrics
        if prices:
            total_unrealized = position_stats.get("total_unrealized_pnl", 0.0)
            total_realized = trading_summary.get("total_realized_pnl", 0.0)
            total_fees = trading_summary.get("total_fees", 0.0)

            combined_stats["combined"] = {
                "total_pnl": total_unrealized + total_realized,
                "net_pnl": total_unrealized + total_realized - total_fees,
                "total_fees": total_fees,
            }

        return combined_stats

    def get_symbol_performance(self, symbol: str, current_price: Optional[float] = None) -> Dict:
        """Get comprehensive performance data for a specific symbol"""
        try:
            # Get current position
            position = self.get_position(symbol)

            # Get trading PnL
            trading_pnl = self.calculate_trading_pnl(symbol)

            # Get recent trades
            recent_trades = self.get_recent_trades(symbol, 5)

            performance = {
                "symbol": symbol,
                "position": {
                    "size": position.size if position else 0.0,
                    "average_price": position.average_price if position else 0.0,
                    "direction": "LONG"
                    if position and position.is_long
                    else "SHORT"
                    if position and position.is_short
                    else "FLAT",
                }
                if position
                else {"size": 0.0, "average_price": 0.0, "direction": "FLAT"},
                "trading_pnl": trading_pnl,
                "recent_trades_count": len(recent_trades),
            }

            # Add unrealized PnL if current price provided
            if current_price and position:
                unrealized_pnl = self.calculate_unrealized_pnl(symbol, current_price)
                performance["unrealized_pnl"] = unrealized_pnl
                performance["current_price"] = current_price

                if position.total_cost > 0:
                    performance["unrealized_pnl_percent"] = (
                        unrealized_pnl / position.total_cost
                    ) * 100

            return performance

        except Exception as e:
            logger.error(f"Failed to get performance for {symbol}: {e}")
            return {"symbol": symbol, "error": str(e)}

    def validate_data_integrity(self) -> bool:
        """Validate all trading data integrity"""
        return self.trade_recorder.validate_trade_integrity()

    def recover_from_trades(self, symbol: Optional[str] = None) -> bool:
        """Recover positions from trade history"""
        return self.trade_recorder.recover_from_trades(symbol)

    def close_position(
        self,
        symbol: str,
        price: float,
        fee: float,
        timestamp: int,
        signal_result: Optional[SignalResult] = None,
    ) -> Optional[Trade]:
        """Close an entire position at market price"""
        position = self.get_position(symbol)
        if not position or position.is_flat:
            logger.warning(f"No position to close for {symbol}")
            return None

        # Determine side and size to close position
        close_side = TradeSide.SELL if position.is_long else TradeSide.BUY
        close_size = abs(position.size)

        logger.info(f"Closing {symbol} position: {close_side.value} {close_size} @ ${price:.4f}")

        return self.record_trade(
            symbol, close_side, price, close_size, fee, timestamp, signal_result
        )

    def get_database_stats(self) -> Dict:
        """Get database statistics"""
        return self.db.get_summary_stats()

    def export_trades_to_dict(self, symbol: Optional[str] = None) -> List[Dict]:
        """Export trade data to dictionary format for analysis"""
        try:
            if symbol:
                trades = self.get_trade_history(symbol)
            else:
                # Get all trades by getting all symbols with positions/trades
                all_symbols = set()
                positions = self.get_all_positions()
                all_symbols.update(positions.keys())

                db_positions = self.db.get_all_positions()
                all_symbols.update(pos.symbol for pos in db_positions)

                trades = []
                for sym in all_symbols:
                    trades.extend(self.get_trade_history(sym))

                # Sort by timestamp
                trades.sort(key=lambda t: t.timestamp)

            # Convert to dict format
            trade_dicts = []
            for trade in trades:
                trade_dict = {
                    "trade_id": trade.trade_id,
                    "timestamp": trade.timestamp,
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "price": trade.price,
                    "size": trade.size,
                    "fee": trade.fee,
                    "total_value": trade.total_value,
                    "net_value": trade.net_value,
                    "signal_metadata": trade.signal_metadata,
                }
                trade_dicts.append(trade_dict)

            return trade_dicts

        except Exception as e:
            logger.error(f"Failed to export trades: {e}")
            return []
