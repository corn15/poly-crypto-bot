import enum
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict


class TradeSide(enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Trade:
    symbol: str
    side: TradeSide
    price: float
    size: float
    fee: float
    timestamp: int
    signal_metadata: Dict[str, Any] = field(default_factory=dict)
    trade_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.size <= 0:
            raise ValueError("Trade size must be positive")
        if self.price <= 0:
            raise ValueError("Trade price must be positive")
        if self.fee < 0:
            raise ValueError("Trade fee cannot be negative")

    @property
    def total_value(self) -> float:
        """Total value of the trade excluding fees"""
        return self.price * self.size

    @property
    def net_value(self) -> float:
        """Net value after fees (positive for buy, negative for sell from cash perspective)"""
        if self.side == TradeSide.BUY:
            return -(self.total_value + self.fee)  # Cash outflow
        else:
            return self.total_value - self.fee  # Cash inflow
