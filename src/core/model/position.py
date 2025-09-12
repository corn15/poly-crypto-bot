from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .trade import Trade


@dataclass
class Position:
    symbol: str
    size: float  # positive for long, negative for short
    average_price: float  # weighted average cost basis
    last_update_ts: int  # milliseconds timestamp

    def __post_init__(self):
        if self.average_price < 0:
            raise ValueError("Average price cannot be negative")
        if self.size != 0 and self.average_price == 0:
            raise ValueError("Non-zero position must have positive average price")

    @property
    def is_long(self) -> bool:
        return self.size > 0

    @property
    def is_short(self) -> bool:
        return self.size < 0

    @property
    def is_flat(self) -> bool:
        return self.size == 0

    @property
    def total_cost(self) -> float:
        """Total cost basis of the position"""
        return abs(self.size) * self.average_price

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL based on current market price"""
        if self.size == 0:
            return 0.0
        return self.size * (current_price - self.average_price)

    def calculate_realized_pnl(self, exit_price: float, exit_size: float) -> float:
        """Calculate realized PnL for a partial or full exit"""
        if self.size == 0:
            return 0.0

        # Ensure exit is in opposite direction
        if (self.size > 0 and exit_size > 0) or (self.size < 0 and exit_size < 0):
            raise ValueError("Exit size must be opposite direction to position")

        exit_size_abs = abs(exit_size)
        max_exit = abs(self.size)

        if exit_size_abs > max_exit:
            raise ValueError(f"Cannot exit {exit_size_abs} from position of {max_exit}")

        return exit_size_abs * (exit_price - self.average_price) * (1 if self.size > 0 else -1)

    def update_from_trade(self, trade: "Trade") -> None:
        """Update position based on a new trade"""
        from .trade import TradeSide

        if trade.symbol != self.symbol:
            raise ValueError(
                f"Trade symbol {trade.symbol} does not match position symbol {self.symbol}"
            )

        trade_size = trade.size if trade.side == TradeSide.BUY else -trade.size

        if self.size == 0:
            # New position
            self.size = trade_size
            self.average_price = trade.price
        elif (self.size > 0 and trade_size > 0) or (self.size < 0 and trade_size < 0):
            # Adding to existing position - update weighted average
            total_cost = self.total_cost + (trade.price * trade.size)
            total_size = abs(self.size) + trade.size
            self.average_price = total_cost / total_size
            self.size += trade_size
        else:
            # Reducing or reversing position
            if abs(trade_size) >= abs(self.size):
                # Full exit or reversal
                remaining_size = abs(trade_size) - abs(self.size)
                if remaining_size > 0:
                    # Reversal - new position in opposite direction
                    self.size = remaining_size * (1 if trade.side == TradeSide.BUY else -1)
                    self.average_price = trade.price
                else:
                    # Flat position
                    self.size = 0
                    self.average_price = 0
            else:
                # Partial exit - keep same average price
                self.size += trade_size

        self.last_update_ts = trade.timestamp
