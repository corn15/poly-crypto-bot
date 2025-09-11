import enum
from dataclasses import dataclass
from typing import Optional


class MarketSignal(enum.Enum):
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"


@dataclass
class SignalResult:
    signal: Optional[MarketSignal]
    probability: Optional[float]
    reasoning: str
    error: Optional[str] = None
