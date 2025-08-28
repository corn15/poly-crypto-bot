from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class SchedulingConfig:
    market_monitor_update_interval: int = 0
    price_feed_update_interval: int = 0

    @classmethod
    def load(cls, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(f"Price schedule config not found at {config_path}")

        try:
            config = cls()
            with open(config_path) as f:
                data = yaml.safe_load(f)
                for key, value in data.items():
                    setattr(config, key, value)
            return config

        except Exception as e:
            raise ValueError(f"Error loading scheduling config from {config_path}: {e}")
