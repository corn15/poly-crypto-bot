import time

import schedule

from src.config.application_config import ApplicationConfig
from src.core.service.market_monitor import MarketMonitor
from src.core.service.price_feed import PriceFeed


class Application:
    def __init__(
        self, config: ApplicationConfig, market_monitor: MarketMonitor, price_feed: PriceFeed
    ):
        self.market_monitor = market_monitor
        self.price_feed = price_feed
        self.config = config

    def run(self):
        market_monitor_update_interval = (
            self.config.scheduling_config.market_monitor_update_interval
        )
        price_feed_update_interval = self.config.scheduling_config.price_feed_update_interval

        print(market_monitor_update_interval)
        print(price_feed_update_interval)
        schedule.every(market_monitor_update_interval).seconds.do(self.market_monitor.update)
        schedule.every(price_feed_update_interval).seconds.do(self.price_feed.update)

        while True:
            schedule.run_pending()
            time.sleep(1)
