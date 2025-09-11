import logging
from datetime import datetime, timezone
from typing import Dict, Tuple

import polars as pl

from src.core.model.prediction_market import UpDown
from src.core.model.price import Candle
from src.core.model.signal import MarketSignal, SignalResult
from src.core.service.market_monitor import MarketMonitorService
from src.core.service.price_feed import PriceFeedService
from src.prediction.predictor import CryptoPredictor

logger = logging.getLogger(__name__)

ASSET_MAPPING = {"bitcoin": "BTCUSDT", "ethereum": "ETHUSDT", "solana": "SOLUSDT"}


class StrategyService:
    def __init__(
        self,
        market_monitor: MarketMonitorService,
        price_provider: PriceFeedService,
        predictor: CryptoPredictor,
    ):
        self.market_monitor = market_monitor
        self.price_provider = price_provider
        self.predictor = predictor

    def generate_signals_for_all_assets(self) -> Dict[str, SignalResult]:
        results = {}
        assets = self.market_monitor.get_assets()

        for asset in assets:
            try:
                result = self.generate_signal_for_asset(asset)
                results[asset] = result

                if result.error:
                    logger.error(
                        "Error processing asset: %s",
                        asset,
                        extra={"asset": asset, "error": result.error},
                    )
                else:
                    log_message = (
                        f"Signal for {asset}: {result.signal.value if result.signal else 'NO_SIGNAL'}\n"
                        f"  Probability: {result.probability:.4f}\n"
                        f"  Reasoning: {result.reasoning}"
                    )
                    logger.info(
                        log_message,
                        extra={
                            "asset": asset,
                            "signal": result.signal.value if result.signal else None,
                            "probability": result.probability,
                            "reasoning": result.reasoning,
                        },
                    )

            except Exception as e:
                logger.exception("Unexpected error processing asset: %s", asset)
                results[asset] = SignalResult(
                    signal=None,
                    probability=None,
                    reasoning="Unexpected error",
                    error=str(e),
                )
        return results

    def generate_signal_for_asset(self, asset: str) -> SignalResult:
        try:
            logger.info(f"Processing asset: {asset}")

            # Get Binance symbol
            exchange_symbol = ASSET_MAPPING.get(asset.lower())
            if not exchange_symbol:
                return SignalResult(
                    signal=None,
                    probability=None,
                    reasoning="Asset mapping not found",
                    error=f"No mapping for {asset}",
                )
            logger.info(f"Fetching candle data for {exchange_symbol}")
            candles = self.price_provider.get_past_4h_5m_candles(exchange_symbol)
            if not candles or len(candles) == 0:
                return SignalResult(
                    signal=None,
                    probability=None,
                    reasoning="Failed to fetch candle data",
                    error="Exchange API unavailable",
                )
            logger.info(f"Fetched {len(candles)} candles for {exchange_symbol}")
            # Convert to DataFrame and predict
            df = self._candles_to_dataframe(candles)
            if df.is_empty():
                return SignalResult(
                    signal=None,
                    probability=None,
                    reasoning="Invalid candle data",
                    error="Data conversion failed",
                )

            probability = self.predictor.predict(df)

            # Apply trading rules
            token1_ask = self.market_monitor.get_best_ask_price(asset.lower(), UpDown.UP)
            token1_bid = self.market_monitor.get_best_bid_price(asset.lower(), UpDown.UP)
            signal, reasoning = self._apply_rules(
                probability,
                token1_ask if token1_ask is not None else float("inf"),
                token1_bid if token1_bid is not None else float("-inf"),
            )

            return SignalResult(
                signal=signal,
                probability=probability,
                reasoning=reasoning,
                error=None,
            )

        except Exception as e:
            return SignalResult(
                signal=None,
                probability=None,
                reasoning="Internal error",
                error=str(e),
            )

    def _candles_to_dataframe(self, candles: list[Candle]) -> pl.DataFrame:
        try:
            data = [
                {
                    "timestamp": candle.open_time,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "open_datetime": datetime.fromtimestamp(
                        candle.open_time / 1000, tz=timezone.utc
                    ),
                }
                for candle in candles
            ]
            return pl.DataFrame(data).sort("timestamp")
        except Exception:
            return pl.DataFrame()

    def _apply_rules(
        self,
        probability: float,
        token1_ask: float,
        token1_bid: float,
    ) -> Tuple[MarketSignal, str]:
        if any(price is None for price in [token1_ask, token1_bid]):
            return MarketSignal.NEUTRAL, "Order book prices unavailable"

        if probability > 0.65 and probability > token1_ask:
            return (
                MarketSignal.BULLISH,
                f"High prob ({probability:.3f}) > token1 ask ({token1_ask:.3f})",
            )

        if probability < 0.35 and probability < token1_bid:
            return (
                MarketSignal.BEARISH,
                f"Low prob ({probability:.3f}) < token1 bid ({token1_bid:.3f})",
            )

        if 0.35 <= probability <= 0.65:
            return MarketSignal.NEUTRAL, f"Neutral prob ({probability:.3f}) in [0.35, 0.65]"

        # Fallback
        return (
            MarketSignal.NEUTRAL,
            f"No rule matched (prob: {probability:.3f}, token1_ask: {token1_ask:.3f}, token1_bid: {token1_bid:.3f})",  # noqa: E501
        )
