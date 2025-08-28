from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class MarketMonitorConfig:
    assets: list[str] = field(default_factory=list)

    initialization_delay_seconds: int = 60

    @classmethod
    def load(cls, config_path: Path):
        if not config_path.exists():
            raise FileNotFoundError(f"Price market monitor config not found at {config_path}")

        try:
            config = cls()
            with open(config_path) as f:
                data = yaml.safe_load(f)
                for key, value in data.items():
                    setattr(config, key, value)
            return config

        except Exception as e:
            raise ValueError(f"Error loading market monitor config from {config_path}: {e}")
