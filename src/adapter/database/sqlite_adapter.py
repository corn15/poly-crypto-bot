import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Optional

from src.core.model.position import Position
from src.core.model.trade import Trade, TradeSide

logger = logging.getLogger(__name__)


class SQLiteAdapter:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database with required tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    timestamp INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    fee REAL NOT NULL,
                    signal_metadata TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    size REAL NOT NULL,
                    average_price REAL NOT NULL,
                    last_update_ts INTEGER NOT NULL
                )
            """)

            # Create indexes for better query performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)")
            conn.commit()

    def insert_trade(self, trade: Trade) -> bool:
        """Insert a new trade record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO trades (trade_id, timestamp, symbol, side, price, size, fee, signal_metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        trade.trade_id,
                        trade.timestamp,
                        trade.symbol,
                        trade.side.value,
                        trade.price,
                        trade.size,
                        trade.fee,
                        json.dumps(trade.signal_metadata),
                    ),
                )
                conn.commit()
                return True
        except sqlite3.IntegrityError:
            logger.warning(f"Trade {trade.trade_id} already exists")
            return False
        except Exception as e:
            logger.error(f"Failed to insert trade: {e}")
            return False

    def get_trade(self, trade_id: str) -> Optional[Trade]:
        """Get a trade by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT trade_id, timestamp, symbol, side, price, size, fee, signal_metadata
                    FROM trades WHERE trade_id = ?
                """,
                    (trade_id,),
                )

                row = cursor.fetchone()
                if row:
                    return Trade(
                        trade_id=row[0],
                        timestamp=row[1],
                        symbol=row[2],
                        side=TradeSide(row[3]),
                        price=row[4],
                        size=row[5],
                        fee=row[6],
                        signal_metadata=json.loads(row[7]),
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get trade {trade_id}: {e}")
            return None

    def get_trades_by_symbol(self, symbol: str, limit: Optional[int] = None) -> List[Trade]:
        """Get trades for a specific symbol, ordered by timestamp desc"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT trade_id, timestamp, symbol, side, price, size, fee, signal_metadata
                    FROM trades WHERE symbol = ? ORDER BY timestamp DESC
                """
                params = [symbol]

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                trades = []

                for row in cursor.fetchall():
                    trades.append(
                        Trade(
                            trade_id=row[0],
                            timestamp=row[1],
                            symbol=row[2],
                            side=TradeSide(row[3]),
                            price=row[4],
                            size=row[5],
                            fee=row[6],
                            signal_metadata=json.loads(row[7]),
                        )
                    )

                return trades
        except Exception as e:
            logger.error(f"Failed to get trades for {symbol}: {e}")
            return []

    def upsert_position(self, position: Position) -> bool:
        """Insert or update a position record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO positions (symbol, size, average_price, last_update_ts)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        position.symbol,
                        position.size,
                        position.average_price,
                        position.last_update_ts,
                    ),
                )
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to upsert position for {position.symbol}: {e}")
            return False

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT symbol, size, average_price, last_update_ts
                    FROM positions WHERE symbol = ?
                """,
                    (symbol,),
                )

                row = cursor.fetchone()
                if row:
                    return Position(
                        symbol=row[0], size=row[1], average_price=row[2], last_update_ts=row[3]
                    )
                return None
        except Exception as e:
            logger.error(f"Failed to get position for {symbol}: {e}")
            return None

    def get_all_positions(self) -> List[Position]:
        """Get all non-zero positions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, size, average_price, last_update_ts
                    FROM positions WHERE size != 0
                """)

                positions = []
                for row in cursor.fetchall():
                    positions.append(
                        Position(
                            symbol=row[0], size=row[1], average_price=row[2], last_update_ts=row[3]
                        )
                    )

                return positions
        except Exception as e:
            logger.error(f"Failed to get all positions: {e}")
            return []

    def delete_position(self, symbol: str) -> bool:
        """Delete a position record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to delete position for {symbol}: {e}")
            return False

    def get_trade_history(
        self, symbol: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None
    ) -> List[Trade]:
        """Get trade history with optional time range filtering"""
        try:
            query = """
                SELECT trade_id, timestamp, symbol, side, price, size, fee, signal_metadata
                FROM trades WHERE symbol = ?
            """
            params = [symbol]

            if start_ts:
                query += " AND timestamp >= ?"
                params.append(start_ts)

            if end_ts:
                query += " AND timestamp <= ?"
                params.append(end_ts)

            query += " ORDER BY timestamp ASC"

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(query, params)
                trades = []

                for row in cursor.fetchall():
                    trades.append(
                        Trade(
                            trade_id=row[0],
                            timestamp=row[1],
                            symbol=row[2],
                            side=TradeSide(row[3]),
                            price=row[4],
                            size=row[5],
                            fee=row[6],
                            signal_metadata=json.loads(row[7]),
                        )
                    )

                return trades
        except Exception as e:
            logger.error(f"Failed to get trade history for {symbol}: {e}")
            return []

    def validate_position_against_trades(self, symbol: str) -> bool:
        """Validate that the stored position matches trade history"""
        try:
            stored_position = self.get_position(symbol)
            trades = self.get_trade_history(symbol)

            if not trades:
                return stored_position is None or stored_position.is_flat

            # Calculate position from trades
            calculated_position = Position(symbol, 0, 0, 0)

            for trade in trades:
                calculated_position.update_from_trade(trade)

            if stored_position is None:
                return calculated_position.is_flat

            # Compare with tolerance for floating point precision
            size_match = abs(stored_position.size - calculated_position.size) < 1e-8
            price_match = (stored_position.is_flat and calculated_position.is_flat) or abs(
                stored_position.average_price - calculated_position.average_price
            ) < 1e-8

            if not (size_match and price_match):
                logger.warning(
                    f"Position mismatch for {symbol}: "
                    f"stored=({stored_position.size}, {stored_position.average_price}) "
                    f"calculated=({calculated_position.size}, {calculated_position.average_price})"
                )
                return False

            return True
        except Exception as e:
            logger.error(f"Failed to validate position for {symbol}: {e}")
            return False

    def get_summary_stats(self) -> dict:
        """Get summary statistics about trades and positions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Trade stats
                trade_cursor = conn.execute("""
                    SELECT COUNT(*), COUNT(DISTINCT symbol),
                           MIN(timestamp), MAX(timestamp)
                    FROM trades
                """)
                trade_row = trade_cursor.fetchone()

                # Position stats
                position_cursor = conn.execute("""
                    SELECT COUNT(*), SUM(CASE WHEN size > 0 THEN 1 ELSE 0 END),
                           SUM(CASE WHEN size < 0 THEN 1 ELSE 0 END)
                    FROM positions WHERE size != 0
                """)
                position_row = position_cursor.fetchone()

                return {
                    "total_trades": trade_row[0] if trade_row[0] else 0,
                    "unique_symbols": trade_row[1] if trade_row[1] else 0,
                    "earliest_trade": trade_row[2],
                    "latest_trade": trade_row[3],
                    "open_positions": position_row[0] if position_row[0] else 0,
                    "long_positions": position_row[1] if position_row[1] else 0,
                    "short_positions": position_row[2] if position_row[2] else 0,
                }
        except Exception as e:
            logger.error(f"Failed to get summary stats: {e}")
            return {}
