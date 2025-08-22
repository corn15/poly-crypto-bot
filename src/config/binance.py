from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class BinanceConfig:
    base_url: str = "https://api.binance.com"
    candle_limit: int = 1000
    candle_interval: str = ""
    assets: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(f"Price feed config file not found at {config_path}")

        try:
            config = cls()
            with open(config_path) as f:
                data = yaml.safe_load(f)
                for key, value in data.items():
                    setattr(config, key, value)
            return config

        except Exception as e:
            raise ValueError(f"Error loading price feed config from {config_path}: {e}")
