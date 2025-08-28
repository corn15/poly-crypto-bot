from typing import Dict, Optional

from src.config.market_monitor_config import MarketMonitorConfig
from src.core.interface.market_client import IMarketClient
from src.core.model.prediction_market import OrderBookPair, UpDown


class MarketMonitor:
    def __init__(self, config: MarketMonitorConfig, market_client: IMarketClient):
        self.market_client: IMarketClient = market_client
        self.order_book_pairs: Dict[str, OrderBookPair] = {}
        self.config = config

    def initialize(self):
        self.update()

    def update(self):
        print("Updating market monitor")
        for a in self.config.assets:
            market = self.market_client.get_current_hourly_market(a)
            if not market:
                continue

            cond1 = market.clob_token_ids[0]
            order_book1 = self.market_client.get_order_book(cond1)

            cond2 = market.clob_token_ids[1]
            order_book2 = self.market_client.get_order_book(cond2)

            if not order_book1 or not order_book2:
                continue

            self.order_book_pairs[a] = OrderBookPair(order_book1, order_book2)

    def get_best_ask_price(self, symbol: str, up_or_down: UpDown) -> Optional[float]:
        if not self.order_book_pairs.get(symbol):
            return None
        if up_or_down == UpDown.UP:
            return self.order_book_pairs[symbol].up.best_ask_price
        elif up_or_down == UpDown.DOWN:
            return self.order_book_pairs[symbol].down.best_bid_price
        else:
            raise ValueError("Invalid up_or_down value")

    def get_best_bid_price(self, symbol: str, up_or_down: UpDown) -> Optional[float]:
        if not self.order_book_pairs.get(symbol):
            return None
        if up_or_down == UpDown.UP:
            return self.order_book_pairs[symbol].up.best_bid_price
        elif up_or_down == UpDown.DOWN:
            return self.order_book_pairs[symbol].down.best_ask_price
        else:
            raise ValueError("Invalid up_or_down value")
