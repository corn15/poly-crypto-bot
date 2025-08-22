from abc import ABC, abstractmethod

from src.core.models.asset import Asset
from src.core.models.prediction_market import OrderBook, PredictionMarket


class IMarketClient(ABC):
    @abstractmethod
    def get_hourly_market(self, asset: Asset) -> PredictionMarket:
        """Retrieve market data for the given asset."""
        pass

    @abstractmethod
    def get_order_book(self, condition_id: str) -> OrderBook:
        """Retrieve market data for the given asset."""
        pass
