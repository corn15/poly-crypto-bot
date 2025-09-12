import logging
from typing import Dict, Optional

from src.adapter.database import SQLiteAdapter
from src.core.model.position import Position
from src.core.model.signal import SignalResult
from src.core.model.trade import Trade, TradeSide
from src.core.service.position_manager import PositionManager

logger = logging.getLogger(__name__)


class TradeRecorder:
    def __init__(self, db_adapter: SQLiteAdapter, position_manager: PositionManager):
        self.db = db_adapter
        self.position_manager = position_manager

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
        """Record a new trade and update position"""
        try:
            # Create trade with signal metadata
            signal_metadata = {}
            if signal_result:
                signal_metadata = {
                    "signal": signal_result.signal.value if signal_result.signal else None,
                    "probability": signal_result.probability,
                    "reasoning": signal_result.reasoning,
                }

            trade = Trade(
                symbol=symbol,
                side=side,
                price=price,
                size=size,
                fee=fee,
                timestamp=timestamp,
                signal_metadata=signal_metadata,
            )

            # Get current position for PnL calculation
            current_position = self.position_manager.get_position(symbol)
            realized_pnl = 0.0

            # Calculate realized PnL before position update
            if current_position and not current_position.is_flat:
                if (current_position.is_long and side == TradeSide.SELL) or (
                    current_position.is_short and side == TradeSide.BUY
                ):
                    try:
                        exit_size = -size if side == TradeSide.SELL else size
                        realized_pnl = current_position.calculate_realized_pnl(price, exit_size)
                    except Exception as e:
                        logger.warning(f"Could not calculate realized PnL for {symbol}: {e}")

            # Record trade in database
            if not self.db.insert_trade(trade):
                logger.error(f"Failed to record trade {trade.trade_id}")
                return None

            # Update position
            if not self.position_manager.update_position_from_trade(trade):
                logger.error(f"Failed to update position from trade {trade.trade_id}")
                return None

            # Log trade details
            direction = "BUY" if side == TradeSide.BUY else "SELL"
            logger.info(
                f"Trade recorded: {direction} {size} {symbol} @ ${price:.4f} "
                f"(fee: ${fee:.4f}, id: {trade.trade_id[:8]})"
            )

            # Log realized PnL if significant
            if abs(realized_pnl) > 0.01:
                logger.info(
                    f"Realized PnL: ${realized_pnl:.4f} "
                    f"(buy: ${current_position.average_price:.4f}, sell: ${price:.4f})"
                )

            return trade

        except Exception as e:
            logger.error(f"Failed to record trade for {symbol}: {e}")
            return None

    def get_trade_history(
        self, symbol: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None
    ) -> list[Trade]:
        """Get trade history for a symbol with optional time filtering"""
        return self.db.get_trade_history(symbol, start_ts, end_ts)

    def get_recent_trades(self, symbol: str, limit: int = 10) -> list[Trade]:
        """Get recent trades for a symbol"""
        return self.db.get_trades_by_symbol(symbol, limit)

    def calculate_trading_pnl(self, symbol: str) -> Dict[str, float]:
        """Calculate trading PnL summary for a symbol"""
        try:
            trades = self.db.get_trade_history(symbol)
            if not trades:
                return {"realized_pnl": 0.0, "total_fees": 0.0, "trade_count": 0}

            # Calculate realized PnL by reconstructing position changes
            temp_position = Position(symbol=symbol, size=0, average_price=0, last_update_ts=0)
            realized_pnl = 0.0
            total_fees = 0.0

            for trade in trades:
                # Calculate realized PnL before updating position
                if not temp_position.is_flat:
                    if (temp_position.is_long and trade.side == TradeSide.SELL) or (
                        temp_position.is_short and trade.side == TradeSide.BUY
                    ):
                        try:
                            exit_size = -trade.size if trade.side == TradeSide.SELL else trade.size
                            pnl = temp_position.calculate_realized_pnl(trade.price, exit_size)
                            realized_pnl += pnl
                        except Exception as e:
                            logger.warning(f"PnL calculation error for trade {trade.trade_id}: {e}")

                # Update temp position
                temp_position.update_from_trade(trade)
                total_fees += trade.fee

            return {
                "realized_pnl": realized_pnl,
                "total_fees": total_fees,
                "trade_count": len(trades),
                "net_pnl": realized_pnl - total_fees,
            }

        except Exception as e:
            logger.error(f"Failed to calculate trading PnL for {symbol}: {e}")
            return {"realized_pnl": 0.0, "total_fees": 0.0, "trade_count": 0}

    def get_trading_summary(self) -> Dict[str, any]:
        """Get overall trading summary across all symbols"""
        try:
            stats = self.db.get_summary_stats()

            # Get all symbols that have been traded
            all_positions = self.position_manager.get_all_positions()
            traded_symbols = set(all_positions.keys())

            # Add any symbols from database that might not have current positions
            db_positions = self.db.get_all_positions()
            for pos in db_positions:
                traded_symbols.add(pos.symbol)

            total_realized_pnl = 0.0
            total_fees = 0.0
            symbol_summaries = {}

            for symbol in traded_symbols:
                pnl_data = self.calculate_trading_pnl(symbol)
                symbol_summaries[symbol] = pnl_data
                total_realized_pnl += pnl_data["realized_pnl"]
                total_fees += pnl_data["total_fees"]

            return {
                "database_stats": stats,
                "total_realized_pnl": total_realized_pnl,
                "total_fees": total_fees,
                "net_trading_pnl": total_realized_pnl - total_fees,
                "symbols_traded": len(traded_symbols),
                "symbol_summaries": symbol_summaries,
            }

        except Exception as e:
            logger.error(f"Failed to get trading summary: {e}")
            return {}

    def validate_trade_integrity(self, symbol: Optional[str] = None) -> bool:
        """Validate trade and position integrity"""
        try:
            if symbol:
                # Validate single symbol
                return self.position_manager.validate_all_positions()
            else:
                # Validate all symbols
                return self.position_manager.validate_all_positions()

        except Exception as e:
            logger.error(f"Failed to validate trade integrity: {e}")
            return False

    def recover_from_trades(self, symbol: Optional[str] = None) -> bool:
        """Recover positions from trade history"""
        return self.position_manager.recover_positions_from_trades(symbol)
