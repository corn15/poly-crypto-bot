from typing import Dict

from src.core.interface.exchange import IExchange
from src.core.model.asset import Asset
from src.core.model.price import Candle


class PriceFeedService:
    def __init__(
        self,
        price_feed_provider: IExchange,
    ):
        self.price_feed_provider = price_feed_provider

        self.price: Dict[str, float] = {}
        self.candles: Dict[str, Candle] = {}

    def update(self):
        for asset in self.price_feed_provider.get_assets():
            self.price[asset] = self.price_feed_provider.get_current_price(asset)
            self.candles[asset] = self.price_feed_provider.get_current_candle(asset)

        print(f"price:{self.price}")

    def get_price(self, asset: Asset) -> float:
        return self.price[asset.symbol]

    def get_candle(self, asset: Asset) -> Candle:
        return self.candles[asset.symbol]
