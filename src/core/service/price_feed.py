from typing import Dict, List

from src.core.interface.exchange import IExchange
from src.core.model.price import Candle


class PriceFeedService:
    def __init__(
        self,
        price_feed_provider: IExchange,
    ):
        self.price_feed_provider = price_feed_provider

        self.price: Dict[str, float] = {}
        self.candles: Dict[str, Candle] = {}
        self.past_3h_5m_candles: Dict[str, List[Candle]] = {}

    def update(self):
        for asset in self.price_feed_provider.get_assets():
            self.price[asset] = self.price_feed_provider.get_current_price(asset)
            self.candles[asset] = self.price_feed_provider.get_current_candle(asset)
            self.past_3h_5m_candles[asset] = self.price_feed_provider.get_historical_candles(
                asset, 36
            )
        print(f"price:{self.price}")

    def get_price(self, asset: str) -> float:
        return self.price[asset]

    def get_candle(self, asset: str) -> Candle:
        return self.candles[asset]

    def get_past_3h_5m_candles(self, asset: str) -> List[Candle]:
        return self.past_3h_5m_candles[asset]
