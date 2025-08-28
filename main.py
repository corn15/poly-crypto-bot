import argparse
from pathlib import Path

from src.adapters.polymarket.market_data_client import MarketDataClient
from src.adapters.price_providers.binance_price_provider import BinancePriceProvider
from src.application.application import Application
from src.config.application_config import ApplicationConfig
from src.core.service.market_monitor import MarketMonitor
from src.core.service.price_feed import PriceFeed


def main():
    parser = argparse.ArgumentParser(description="Polymarket Crypto Expiry Prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    config = ApplicationConfig.load(Path(args.config))
    print(config)
    binance_price_provider = BinancePriceProvider(config.price_feed_config)
    market_data_client = MarketDataClient()
    market_monitor_service = MarketMonitor(config.market_monitor_config, market_data_client)
    price_feed_service = PriceFeed(config.price_feed_config, binance_price_provider)

    app = Application(config, market_monitor_service, price_feed_service)
    app.run()


if __name__ == "__main__":
    main()
