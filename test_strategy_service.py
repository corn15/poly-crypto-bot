import logging
from pathlib import Path

from src.adapter.exchange.binance import Binance
from src.adapter.prediction_market.polymarket import Polymarket
from src.application.strategy_service import StrategyService
from src.config.application import ApplicationConfig
from src.prediction.predictor import CryptoPredictor

logging.basicConfig(level=logging.INFO)


def test_strategy_service():
    """Test StrategyService functionality"""
    print("Testing StrategyService...")

    try:
        # Initialize
        config = ApplicationConfig.load(Path("config/config.yaml"))
        binance = Binance(config.binance)
        polymarket = Polymarket(config.polymarket)
        predictor = CryptoPredictor("models")

        strategy_service = StrategyService(binance, polymarket, predictor)
        print("✅ Initialization successful")

        # Test single asset
        result = strategy_service.generate_signal_for_asset("bitcoin")
        print(f"✅ Single asset test: {result.signal}")

        # Test all assets
        results = strategy_service.generate_signals_for_all_assets()
        print(f"✅ All assets test: {len(results)} results")

        for asset, result in results.items():
            status = "✅" if not result.error else "❌"
            signal = result.signal.value if result.signal else "NO_SIGNAL"
            print(f"  {status} {asset}: {signal}")

        print("✅ Test completed successfully")

    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    test_strategy_service()
