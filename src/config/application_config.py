from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.config.market_monitor_config import MarketMonitorConfig
from src.config.price_feed_config import PriceFeedConfig
from src.config.scheduling_config import SchedulingConfig


@dataclass
class ApplicationConfig:
    market_monitor_config: MarketMonitorConfig = field(default_factory=MarketMonitorConfig)
    price_feed_config: PriceFeedConfig = field(default_factory=PriceFeedConfig)
    scheduling_config: SchedulingConfig = field(default_factory=SchedulingConfig)

    @classmethod
    def load(cls, config_path: Path) -> "ApplicationConfig":
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        try:
            config = cls()
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            if data:
                if "market_monitor" in data:
                    config.market_monitor_config = MarketMonitorConfig(**data["market_monitor"])
                if "price_feed" in data:
                    config.price_feed_config = PriceFeedConfig(**data["price_feed"])
                if "scheduling" in data:
                    config.scheduling_config = SchedulingConfig(**data["scheduling"])
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")
