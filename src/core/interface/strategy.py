from abc import ABC, abstractmethod

from src.core.model.price import Candle
from src.core.model.signal import Signal


class IStrategy(ABC):
    @abstractmethod
    def generate_signal(self, candles: list[Candle], current_price: float) -> Signal:
        pass
