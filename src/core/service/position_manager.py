import logging
from typing import Dict, Optional

from src.adapter.database import SQLiteAdapter
from src.core.model.position import Position
from src.core.model.trade import Trade

logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(self, db_adapter: SQLiteAdapter):
        self.db = db_adapter
        self._positions: Dict[str, Position] = {}
        self._load_positions()

    def _load_positions(self) -> None:
        """Load all open positions from database"""
        try:
            positions = self.db.get_all_positions()
            for position in positions:
                self._positions[position.symbol] = position
            logger.info(f"Loaded {len(positions)} open positions")
        except Exception as e:
            logger.error(f"Failed to load positions: {e}")
            self._positions = {}

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol"""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, Position]:
        """Get all current positions"""
        return self._positions.copy()

    def update_position_from_trade(self, trade: Trade) -> bool:
        """Update position based on a new trade and persist to database"""
        try:
            symbol = trade.symbol

            # Get or create position
            if symbol in self._positions:
                position = self._positions[symbol]
            else:
                position = Position(symbol=symbol, size=0, average_price=0, last_update_ts=0)
                self._positions[symbol] = position

            # Calculate realized PnL before position update
            realized_pnl = 0.0
            if not position.is_flat:
                try:
                    from src.core.model.trade import TradeSide

                    if (position.is_long and trade.side == TradeSide.SELL) or (
                        position.is_short and trade.side == TradeSide.BUY
                    ):
                        exit_size = trade.size if trade.side == TradeSide.SELL else -trade.size
                        realized_pnl = position.calculate_realized_pnl(trade.price, exit_size)
                except Exception as e:
                    logger.warning(f"Could not calculate realized PnL for {symbol}: {e}")

            # Log realized PnL if significant
            if abs(realized_pnl) > 0.01:  # Log if > 1 cent
                logger.info(
                    f"Realized PnL for {symbol}: ${realized_pnl:.4f} "
                    f"(buy_price: ${position.average_price:.4f}, "
                    f"sell_price: ${trade.price:.4f}, "
                    f"size: {trade.size})"
                )

            # Update position
            position.update_from_trade(trade)

            # Remove from memory if position is flat
            if position.is_flat:
                if symbol in self._positions:
                    del self._positions[symbol]
                # Delete from database as well
                self.db.delete_position(symbol)
            else:
                # Persist to database
                success = self.db.upsert_position(position)
                if not success:
                    logger.error(f"Failed to persist position for {symbol}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Failed to update position from trade {trade.trade_id}: {e}")
            return False

    def calculate_unrealized_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized PnL for a specific symbol"""
        position = self.get_position(symbol)
        if position is None or position.is_flat:
            return 0.0

        try:
            return position.calculate_unrealized_pnl(current_price)
        except Exception as e:
            logger.error(f"Failed to calculate unrealized PnL for {symbol}: {e}")
            return 0.0

    def calculate_total_unrealized_pnl(self, prices: Dict[str, float]) -> float:
        """Calculate total unrealized PnL across all positions"""
        total_pnl = 0.0

        for symbol, position in self._positions.items():
            if symbol in prices:
                try:
                    pnl = position.calculate_unrealized_pnl(prices[symbol])
                    total_pnl += pnl
                except Exception as e:
                    logger.error(f"Error calculating PnL for {symbol}: {e}")
            else:
                logger.warning(f"No price available for {symbol}")

        return total_pnl

    def get_position_summary(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, Dict]:
        """Get summary of all positions with optional PnL calculations"""
        summary = {}

        for symbol, position in self._positions.items():
            pos_info = {
                "symbol": symbol,
                "size": position.size,
                "average_price": position.average_price,
                "total_cost": position.total_cost,
                "direction": "LONG" if position.is_long else "SHORT",
                "last_update_ts": position.last_update_ts,
            }

            if prices and symbol in prices:
                current_price = prices[symbol]
                unrealized_pnl = position.calculate_unrealized_pnl(current_price)
                pos_info.update(
                    {
                        "current_price": current_price,
                        "unrealized_pnl": unrealized_pnl,
                        "pnl_percent": (unrealized_pnl / position.total_cost) * 100
                        if position.total_cost > 0
                        else 0,
                    }
                )

            summary[symbol] = pos_info

        return summary

    def validate_all_positions(self) -> bool:
        """Validate all positions against trade history"""
        try:
            all_symbols = set(self._positions.keys())

            # Also check symbols that might have trades but no position
            all_positions_in_db = self.db.get_all_positions()
            for pos in all_positions_in_db:
                all_symbols.add(pos.symbol)

            validation_results = {}
            for symbol in all_symbols:
                is_valid = self.db.validate_position_against_trades(symbol)
                validation_results[symbol] = is_valid

                if not is_valid:
                    logger.error(f"Position validation failed for {symbol}")

            valid_count = sum(validation_results.values())
            total_count = len(validation_results)

            logger.info(f"Position validation: {valid_count}/{total_count} positions valid")
            return valid_count == total_count

        except Exception as e:
            logger.error(f"Failed to validate positions: {e}")
            return False

    def recover_positions_from_trades(self, symbol: Optional[str] = None) -> bool:
        """Recover positions by recalculating from trade history"""
        try:
            symbols_to_recover = [symbol] if symbol else list(self._positions.keys())

            # Also recover any symbol that has trades but no position
            if not symbol:
                all_positions_in_db = self.db.get_all_positions()
                for pos in all_positions_in_db:
                    if pos.symbol not in symbols_to_recover:
                        symbols_to_recover.append(pos.symbol)

            recovered_count = 0
            for sym in symbols_to_recover:
                if self._recover_single_position(sym):
                    recovered_count += 1

            logger.info(f"Recovered {recovered_count}/{len(symbols_to_recover)} positions")
            return recovered_count == len(symbols_to_recover)

        except Exception as e:
            logger.error(f"Failed to recover positions: {e}")
            return False

    def _recover_single_position(self, symbol: str) -> bool:
        """Recover a single position from its trade history"""
        try:
            trades = self.db.get_trade_history(symbol)
            if not trades:
                # No trades, position should be flat
                if symbol in self._positions:
                    del self._positions[symbol]
                self.db.delete_position(symbol)
                return True

            # Recalculate position from trades
            recovered_position = Position(symbol=symbol, size=0, average_price=0, last_update_ts=0)

            for trade in trades:
                recovered_position.update_from_trade(trade)

            # Update memory and database
            if recovered_position.is_flat:
                if symbol in self._positions:
                    del self._positions[symbol]
                self.db.delete_position(symbol)
            else:
                self._positions[symbol] = recovered_position
                self.db.upsert_position(recovered_position)

            logger.info(
                f"Recovered position for {symbol}: size={recovered_position.size}, avg_price={recovered_position.average_price}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to recover position for {symbol}: {e}")
            return False

    def get_portfolio_stats(self, prices: Optional[Dict[str, float]] = None) -> Dict:
        """Get portfolio-level statistics"""
        try:
            total_positions = len(self._positions)
            long_positions = sum(1 for pos in self._positions.values() if pos.is_long)
            short_positions = sum(1 for pos in self._positions.values() if pos.is_short)

            total_cost = sum(pos.total_cost for pos in self._positions.values())

            stats = {
                "total_positions": total_positions,
                "long_positions": long_positions,
                "short_positions": short_positions,
                "total_cost_basis": total_cost,
            }

            if prices:
                total_unrealized_pnl = self.calculate_total_unrealized_pnl(prices)
                stats.update(
                    {
                        "total_unrealized_pnl": total_unrealized_pnl,
                        "portfolio_return_percent": (total_unrealized_pnl / total_cost) * 100
                        if total_cost > 0
                        else 0,
                    }
                )

            return stats

        except Exception as e:
            logger.error(f"Failed to get portfolio stats: {e}")
            return {}
