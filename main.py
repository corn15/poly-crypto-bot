import argparse
from pathlib import Path

from src.adapter.exchange.binance import Binance
from src.adapter.prediction_market.polymarket import Polymarket
from src.application.application import Application
from src.config.application import ApplicationConfig
from src.core.service.market_monitor import MarketMonitorService
from src.core.service.price_feed import PriceFeedService
from src.core.service.strategy import StrategyService
from src.prediction.predictor import CryptoPredictor


def main():
    parser = argparse.ArgumentParser(description="Polymarket Crypto Expiry Prediction")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Load configuration
    config = ApplicationConfig.load(Path(args.config))
    print(config)

    # Initialize data providers
    binance_price_provider = Binance(config.binance)
    market_data_client = Polymarket(config.polymarket)
    predictor = CryptoPredictor("models")

    # Initialize services
    price_feed_service = PriceFeedService(binance_price_provider)
    market_monitor_service = MarketMonitorService(market_data_client)
    strategy_service = StrategyService(market_monitor_service, price_feed_service, predictor)

    # Start application
    app = Application(config, market_monitor_service, price_feed_service, strategy_service)
    app.run()


if __name__ == "__main__":
    main()
