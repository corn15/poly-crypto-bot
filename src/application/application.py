import time

import schedule

from src.config.application import ApplicationConfig
from src.core.service.market_monitor import MarketMonitorService
from src.core.service.price_feed import PriceFeedService
from src.core.service.strategy import StrategyService


class Application:
    def __init__(
        self,
        config: ApplicationConfig,
        market_monitor: MarketMonitorService,
        price_feed: PriceFeedService,
        strategy_service: StrategyService,
    ):
        self.market_monitor = market_monitor
        self.price_feed = price_feed
        self.strategy_service = strategy_service
        self.config = config

    def run(self):
        market_monitor_update_interval = self.config.scheduling.market_monitor_update_interval
        price_feed_update_interval = self.config.scheduling.price_feed_update_interval

        # Schedule tasks
        schedule.every(market_monitor_update_interval).seconds.do(self.market_monitor.update)
        schedule.every(price_feed_update_interval).seconds.do(self.price_feed.update)
        schedule.every(15).seconds.do(self.strategy_service.generate_signals_for_all_assets)

        # Run scheduler
        while True:
            schedule.run_pending()
            time.sleep(1)
