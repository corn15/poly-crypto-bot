from abc import ABC, abstractmethod

from src.core.model.price import Candle


class IExchange(ABC):
    @abstractmethod
    def get_assets(self) -> list[str]:
        pass

    @abstractmethod
    def get_current_price(self, asset: str) -> float:
        pass

    @abstractmethod
    def get_current_candle(self, asset: str) -> Candle:
        pass

    @abstractmethod
    def get_historical_candles(self, asset: str, limit: int) -> list[Candle]:
        pass
