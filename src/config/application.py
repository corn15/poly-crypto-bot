from dataclasses import dataclass, field
from pathlib import Path

import yaml

from src.config.binance import BinanceConfig
from src.config.polymarket import PolymarketConfig
from src.config.scheduling import SchedulingConfig


@dataclass
class ApplicationConfig:
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    polymarket: PolymarketConfig = field(default_factory=PolymarketConfig)
    scheduling: SchedulingConfig = field(default_factory=SchedulingConfig)

    @classmethod
    def load(cls, config_path: Path) -> "ApplicationConfig":
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        try:
            config = cls()
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            if data:
                if "polymarket" in data:
                    config.polymarket = PolymarketConfig(**data["polymarket"])
                if "binance" in data:
                    config.binance = BinanceConfig(**data["binance"])
                if "scheduling" in data:
                    config.scheduling = SchedulingConfig(**data["scheduling"])
            return config
        except Exception as e:
            raise ValueError(f"Failed to load config file: {e}")
