import logging
import warnings

import polars as pl

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Define feature columns
FEATURES = [
    "minute_in_hour",
    "ret_5m",
    "ret_15m",
    "ret_30m",
    "ret_1h",
    "ret_2h",
    "ret_3h",
    "price_deviation_from_open",
    "relative_price_position",
    "vol_share_in_hour",
    "em5_ratio",
    "em10_ratio",
    "em20_ratio",
    "em30_ratio",
    "ema5_ema10_ratio",
    "ema5_ema20_ratio",
    "ema5_ema30_ratio",
    "ema10_ema20_ratio",
    "ema10_ema30_ratio",
    "ema20_ema30_ratio",
    "rsi_5",
    "rsi_10",
    "rsi_20",
    "rsi_30",
]


def calculate_ema(df: pl.DataFrame, column: str, span: int) -> pl.DataFrame:
    """Calculate Exponential Moving Average."""
    alpha = 2.0 / (span + 1.0)
    return df.with_columns([pl.col(column).ewm_mean(alpha=alpha).alias(f"ema_{span}")])


def calculate_rsi(df: pl.DataFrame, column: str, period: int) -> pl.DataFrame:
    """Calculate Relative Strength Index."""
    delta = df.select(pl.col(column).diff().alias("delta"))["delta"]

    # Convert to pandas for RSI calculation (polars doesn't have built-in RSI)
    delta_pd = delta.to_pandas()

    gain = delta_pd.where(delta_pd > 0, 0).rolling(window=period).mean()
    loss = (-delta_pd.where(delta_pd < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return df.with_columns([pl.Series(f"rsi_{period}", rsi.values)])  # type: ignore


def create_features(df: pl.DataFrame) -> pl.DataFrame:
    """Create all required features for the model."""
    logger.info("Starting feature engineering...")

    # Sort by time to ensure proper calculation
    df = df.sort("open_datetime")

    # Basic time features
    df = df.with_columns(
        [
            pl.col("open_datetime").dt.minute().alias("minute"),
            pl.col("open_datetime").dt.hour().alias("hour"),
        ]
    )

    # minute_in_hour: Proportion of minutes passed within the current hour (0 to 1)
    df = df.with_columns([(pl.col("minute") / 60.0).alias("minute_in_hour")])

    # Calculate log returns for various periods
    df = df.with_columns(
        [
            # ret_5m: Log return over the last 5 minutes (1 period for 5m data)
            (pl.col("close") / pl.col("close").shift(1)).log().alias("ret_5m"),
            # ret_15m: Log return over the last 15 minutes (3 periods)
            (pl.col("close") / pl.col("close").shift(3)).log().alias("ret_15m"),
            # ret_30m: Log return over the last 30 minutes (6 periods)
            (pl.col("close") / pl.col("close").shift(6)).log().alias("ret_30m"),
            # ret_1h: Log return over the last 1 hour (12 periods)
            (pl.col("close") / pl.col("close").shift(12)).log().alias("ret_1h"),
            # ret_2h: Log return over the last 2 hours (24 periods)
            (pl.col("close") / pl.col("close").shift(24)).log().alias("ret_2h"),
            # ret_3h: Log return over the last 3 hours (36 periods)
            (pl.col("close") / pl.col("close").shift(36)).log().alias("ret_3h"),
        ]
    )

    # Calculate hourly aggregates for price deviation and relative position
    df = df.with_columns([pl.col("open_datetime").dt.truncate("1h").alias("hour_start")])

    # Calculate hourly open, high, low for each hour
    hourly_agg = df.group_by("hour_start").agg(
        [
            pl.col("open").first().alias("hourly_open"),
            pl.col("high").max().alias("hourly_high"),
            pl.col("low").min().alias("hourly_low"),
            pl.col("volume").sum().alias("hourly_volume")
            if "volume" in df.columns
            else pl.lit(0).alias("hourly_volume"),
        ]
    )

    df = df.join(hourly_agg, on="hour_start", how="left")

    # price_deviation_from_open: (Current price - Hourly open price) / Hourly open price
    df = df.with_columns(
        [
            ((pl.col("close") - pl.col("hourly_open")) / pl.col("hourly_open")).alias(
                "price_deviation_from_open"
            )
        ]
    )

    # relative_price_position: (Current price - Hourly low) / (Hourly high - Hourly low)
    df = df.with_columns(
        [
            (
                (pl.col("close") - pl.col("hourly_low"))
                / (pl.col("hourly_high") - pl.col("hourly_low"))
            ).alias("relative_price_position")
        ]
    )

    # vol_share_in_hour:
    # Current hour's accumulated volume / Average hourly volume over the past 3 hours
    if "volume" in df.columns:
        avg_hourly_vol = df.select(
            [pl.col("hourly_volume").rolling_mean(window_size=3).alias("avg_hourly_volume_3h")]
        )["avg_hourly_volume_3h"]

        df = df.with_columns(
            [
                pl.Series("avg_hourly_volume_3h", avg_hourly_vol),
                (pl.col("hourly_volume") / avg_hourly_vol).alias("vol_share_in_hour"),
            ]
        )
    else:
        df = df.with_columns([pl.lit(0).alias("vol_share_in_hour")])

    # Calculate EMAs
    for span in [5, 10, 20, 30]:
        df = calculate_ema(df, "close", span)

    # EMA ratios
    df = df.with_columns(
        [
            # em5_ratio: Current price / EMA-5
            (pl.col("close") / pl.col("ema_5")).alias("em5_ratio"),
            # em10_ratio: Current price / EMA-10
            (pl.col("close") / pl.col("ema_10")).alias("em10_ratio"),
            # em20_ratio: Current price / EMA-20
            (pl.col("close") / pl.col("ema_20")).alias("em20_ratio"),
            # em30_ratio: Current price / EMA-30
            (pl.col("close") / pl.col("ema_30")).alias("em30_ratio"),
            # EMA cross ratios
            (pl.col("ema_5") / pl.col("ema_10")).alias("ema5_ema10_ratio"),
            (pl.col("ema_5") / pl.col("ema_20")).alias("ema5_ema20_ratio"),
            (pl.col("ema_5") / pl.col("ema_30")).alias("ema5_ema30_ratio"),
            (pl.col("ema_10") / pl.col("ema_20")).alias("ema10_ema20_ratio"),
            (pl.col("ema_10") / pl.col("ema_30")).alias("ema10_ema30_ratio"),
            (pl.col("ema_20") / pl.col("ema_30")).alias("ema20_ema30_ratio"),
        ]
    )

    # Calculate RSI for different periods
    for period in [5, 10, 20, 30]:
        df = calculate_rsi(df, "close", period)

    logger.info(f"Created {len(FEATURES)} features")
    return df
