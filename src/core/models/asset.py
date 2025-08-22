from dataclasses import dataclass


@dataclass
class Asset:
    id: str
    name: str
    symbol: str
    polymarket_name: str
    binance_symbol: str
