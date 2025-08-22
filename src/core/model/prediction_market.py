from dataclasses import dataclass
from typing import Optional


class UpDown:
    UP = "UP"
    DOWN = "DOWN"


@dataclass
class PredictionMarket:
    id: str
    question: str
    condition_id: str
    slug: str
    description: str
    outcomes: list[str]
    clob_token_ids: list[str]


@dataclass
class Order:
    price: float
    quantity: float


@dataclass
class OrderBook:
    market_id: str
    asset_id: str
    timestamp: int
    asks: list[Order]
    bids: list[Order]
    tick_size: float
    min_order_size: float
    best_ask_price: Optional[float] = None
    best_bid_price: Optional[float] = None

    def __post_init__(self):
        max_bid_price = max(bid.price for bid in self.bids) if self.bids else None
        min_ask_price = min(ask.price for ask in self.asks) if self.asks else None
        self.best_ask_price = min_ask_price
        self.best_bid_price = max_bid_price


@dataclass
class OrderBookPair:
    up: OrderBook
    down: OrderBook
