from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PolymarketConfig:
    assets: list[str] = field(default_factory=list)
    gamma_api_base_url: str = "https://gamma-api.polymarket.com"
    clob_base_url: str = "https://clob.polymarket.com"
    base_url: str = "https://polymarket.com/api"


@classmethod
def load(cls, config_path: Path):
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return cls(**config)
