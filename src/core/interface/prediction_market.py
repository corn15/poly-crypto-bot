from abc import ABC, abstractmethod

from typing_extensions import Optional

from src.core.model.prediction_market import OrderBook, PredictionMarket


class IPredictionMarket(ABC):
    @abstractmethod
    def get_assets(self) -> list[str]:
        pass

    @abstractmethod
    def get_current_hourly_market(self, asset: str) -> Optional[PredictionMarket]:
        """Retrieve market data for the given asset."""
        pass

    @abstractmethod
    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Retrieve market data for the given asset."""
        pass

    @abstractmethod
    def get_current_hourly_open_price(self, asset: str) -> Optional[float]:
        """Retrieve the hourly open price for the given asset."""
        pass
