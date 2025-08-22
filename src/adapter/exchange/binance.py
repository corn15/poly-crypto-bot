import httpx

from src.config.binance import BinanceConfig
from src.core.interface.exchange import IExchange
from src.core.model.price import Candle


class Binance(IExchange):
    def __init__(self, config: BinanceConfig):
        self.config = config
        self.base_url = config.base_url

    def get_assets(self) -> list[str]:
        assets = []
        for asset in self.config.assets:
            assets.append(asset)
        return assets

    def get_current_price(self, asset: str) -> float:
        with httpx.Client() as client:
            url = f"{self.base_url}/api/v3/ticker/price"
            params = {"symbol": asset}
            response = client.get(url, params=params)
            response.raise_for_status()
            return float(response.json()["price"])

    def get_current_candle(self, asset: str) -> Candle:
        with httpx.Client() as client:
            url = f"{self.base_url}/api/v3/klines"
            params = {"symbol": asset, "interval": self.config.candle_interval}
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            latest = data[-1]
            return Candle(
                open=float(latest[1]),
                high=float(latest[2]),
                low=float(latest[3]),
                close=float(latest[4]),
                volume=float(latest[5]),
                open_time=latest[0],
                close_time=latest[6],
            )

    def get_historical_candles(self, asset: str, limit: int) -> list[Candle]:
        with httpx.Client() as client:
            url = f"{self.base_url}/api/v3/klines"
            params = {"symbol": asset, "interval": self.config.candle_interval, "limit": limit}
            response = client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            candles = [
                Candle(
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    open_time=candle[0],
                    close_time=candle[6],
                )
                for candle in data
            ]

            return candles
