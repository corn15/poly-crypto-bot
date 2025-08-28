from typing import Dict

from src.config.price_feed_config import PriceFeedConfig
from src.core.interface.price_data_provider import IPriceDataProvider
from src.core.model.asset import Asset
from src.core.model.price import Candle


class PriceFeed:
    def __init__(
        self,
        config: PriceFeedConfig,
        price_feed_provider: IPriceDataProvider,
    ):
        self.price_feed_provider = price_feed_provider
        self.config = config

        self.price: Dict[str, float] = {}
        self.candles: Dict[str, Candle] = {}

    def update(self):
        for asset in self.config.assets:
            self.price[asset] = self.price_feed_provider.get_current_price(asset)
            self.candles[asset] = self.price_feed_provider.get_current_candle(asset)

    def get_price(self, asset: Asset) -> float:
        return self.price[asset.symbol]

    def get_candle(self, asset: Asset) -> Candle:
        return self.candles[asset.symbol]
