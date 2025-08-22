from src.core.interfaces.market_client import IMarketClient
from src.core.models.asset import Asset
from src.core.models.prediction_market import PredictionMarket


# update order books and delete expired markets
class MarketMonitor:
    def __init__(self, config: dict, market_client: IMarketClient):
        self.market_client = market_client
        self.order_books = {}

    def get_hourly_market(self, asset: Asset) -> PredictionMarket:
        return self.market_client.get_hourly_market(asset)

    def get_best_ask_price(self, token_id: str) -> float:
        if self.order_books.get(token_id):
            return self.order_books[token_id].best_ask_price
        else:
            return None

    def get_best_bid_price(self, token_id: str) -> float:
        if self.order_books.get(token_id):
            return self.order_books[token_id].best_bid_price
        else:
            return None
