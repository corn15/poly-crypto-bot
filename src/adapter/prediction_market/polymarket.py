import json
import logging
import time
from datetime import datetime

import httpx
from typing_extensions import Optional

from src.config.polymarket import PolymarketConfig
from src.core.interface.prediction_market import IPredictionMarket
from src.core.model.prediction_market import Order, OrderBook, PredictionMarket
from src.utils.time import get_time_str

# This will silence the INFO logs from httpx, including the request/response logs
logging.getLogger("httpx").setLevel(logging.WARNING)


class Polymarket(IPredictionMarket):
    def __init__(self, config: PolymarketConfig):
        self.gamma_api_base_url = config.gamma_api_base_url
        self.clob_base_url = config.clob_base_url
        self.base_url = config.base_url
        self.config = config

    def get_assets(self) -> list[str]:
        assets = []
        for asset in self.config.assets:
            assets.append(asset)
        return assets

    def get_current_hourly_market(self, asset: str) -> Optional[PredictionMarket]:
        slug = f"{asset}-up-or-down-{get_time_str(datetime.fromtimestamp(time.time()))}"
        url = f"{self.gamma_api_base_url}/markets/slug/{slug}"
        try:
            with httpx.Client() as client:
                response = client.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if not data:
                    return None
                outcomes = json.loads(data.get("outcomes", []))
                clob_token_ids = json.loads(data.get("clobTokenIds", []))
                market = PredictionMarket(
                    data["id"],
                    data["question"],
                    data["conditionId"],
                    data["slug"],
                    data["description"],
                    outcomes,
                    clob_token_ids,
                )
                return market

        except Exception as e:
            print(f"❌ Failed to get {asset} market : {e}")
            return None

    def get_order_book(self, token_id: str) -> Optional[OrderBook]:
        url = f"{self.clob_base_url}/book"
        params = {"token_id": token_id}
        try:
            with httpx.Client() as client:
                response = client.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if not data:
                    return None
                asks = [Order(float(ask["price"]), float(ask["size"])) for ask in data["asks"]]
                bids = [Order(float(bid["price"]), float(bid["size"])) for bid in data["bids"]]
                order_book = OrderBook(
                    data["market"],
                    data["asset_id"],
                    data["timestamp"],
                    asks,
                    bids,
                    data["tick_size"],
                    data["min_order_size"],
                )
                return order_book

        except Exception as e:
            print(f"❌ Failed to get order book for {token_id}: {e}")
            return None

    def get_current_hourly_open_price(self, asset: str) -> Optional[float]:
        now = time.strftime("%Y-%m-%dT%H:00:00Z", time.gmtime())
        url = f"{self.base_url}/crypto/crypto-price"
        params = {
            "symbol": asset,
            "variant": "hourly",
            "eventStartTime": now,
            "eventEndDate": now,
        }
        try:
            with httpx.Client() as client:
                response = client.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if not data:
                    return None
                return data["openPrice"]

        except Exception as e:
            print(f"❌ Failed to get current hourly open price for {asset}: {e}")
            return None
