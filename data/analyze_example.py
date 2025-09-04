#!/usr/bin/env python3
"""
Example script showing how to analyze downloaded candlestick data.

This script demonstrates:
- Loading data from both JSON and Parquet formats
- Basic data analysis and statistics
- Creating simple technical indicators
- Data visualization examples
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_json_data(filepath: Path) -> pd.DataFrame:
    """Load candlestick data from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    # Convert timestamps to datetime
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    return df


def load_parquet_data(filepath: Path) -> pd.DataFrame:
    """Load candlestick data from Parquet file."""
    return pd.read_parquet(filepath)


def calculate_basic_stats(df: pd.DataFrame) -> dict:
    """Calculate basic statistics for the candlestick data."""
    return {
        "total_candles": len(df),
        "date_range": {
            "start": df["open_time"].min(),
            "end": df["close_time"].max(),
        },
        "price_stats": {
            "min_price": df[["open", "high", "low", "close"]].min().min(),
            "max_price": df[["open", "high", "low", "close"]].max().max(),
            "avg_close": df["close"].mean(),
            "price_volatility": df["close"].std(),
        },
        "volume_stats": {
            "total_volume": df.get("volume", pd.Series([0])).sum(),
            "avg_volume": df.get("volume", pd.Series([0])).mean(),
        },
    }


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add common technical indicators to the dataframe."""
    df = df.copy()

    # Price change and returns
    df["price_change"] = df["close"] - df["open"]
    df["price_change_pct"] = (df["close"] - df["open"]) / df["open"] * 100
    df["returns"] = df["close"].pct_change() * 100

    # Moving averages
    df["sma_20"] = df["close"].rolling(window=20).mean()
    df["sma_50"] = df["close"].rolling(window=50).mean()
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()

    # MACD
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Bollinger Bands
    df["bb_middle"] = df["close"].rolling(window=20).mean()
    bb_std = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_middle"] + (bb_std * 2)
    df["bb_lower"] = df["bb_middle"] - (bb_std * 2)

    # RSI (Relative Strength Index)
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # High-Low spread
    df["hl_spread"] = df["high"] - df["low"]
    df["hl_spread_pct"] = (df["high"] - df["low"]) / df["close"] * 100

    return df


def analyze_price_patterns(df: pd.DataFrame) -> dict:
    """Analyze common price patterns in the data."""
    patterns = {}

    # Bullish vs Bearish candles
    bullish = df["close"] > df["open"]
    patterns["bullish_candles"] = bullish.sum()
    patterns["bearish_candles"] = (~bullish).sum()
    patterns["bullish_ratio"] = bullish.mean()

    # Doji patterns (open ≈ close)
    doji_threshold = df["close"].std() * 0.1  # 10% of price std as threshold
    doji = abs(df["close"] - df["open"]) <= doji_threshold
    patterns["doji_candles"] = doji.sum()
    patterns["doji_ratio"] = doji.mean()

    # Long wicks (high/low extend significantly beyond open/close)
    body_size = abs(df["close"] - df["open"])
    upper_wick = df["high"] - df[["open", "close"]].max(axis=1)
    lower_wick = df[["open", "close"]].min(axis=1) - df["low"]

    patterns["avg_body_size"] = body_size.mean()
    patterns["avg_upper_wick"] = upper_wick.mean()
    patterns["avg_lower_wick"] = lower_wick.mean()

    # Hammer patterns (long lower wick, small body)
    hammer = (lower_wick > body_size * 2) & (upper_wick < body_size * 0.5)
    patterns["hammer_candles"] = hammer.sum()

    return patterns


def create_visualizations(
    df: pd.DataFrame, pair_name: str, timeframe: str, output_dir: Path
) -> None:
    """Create visualization plots for the data."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up the plotting style
    plt.style.use("seaborn-v0_8")
    fig_size = (15, 10)

    # 1. Price Chart with Moving Averages
    plt.figure(figsize=fig_size)
    plt.subplot(2, 2, 1)
    plt.plot(df.index, df["close"], label="Close Price", alpha=0.7)
    if "sma_20" in df.columns:
        plt.plot(df.index, df["sma_20"], label="SMA 20", alpha=0.8)
        plt.plot(df.index, df["sma_50"], label="SMA 50", alpha=0.8)
    plt.title(f"{pair_name} {timeframe} - Price Chart")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Volume Chart (if available)
    plt.subplot(2, 2, 2)
    if "volume" in df.columns and not df["volume"].isna().all():
        plt.bar(df.index, df["volume"], alpha=0.6)
        plt.title(f"{pair_name} {timeframe} - Volume")
        plt.xlabel("Time")
        plt.ylabel("Volume")
    else:
        plt.bar(df.index, df["hl_spread"], alpha=0.6, color="orange")
        plt.title(f"{pair_name} {timeframe} - High-Low Spread")
        plt.xlabel("Time")
        plt.ylabel("Price Spread")
    plt.grid(True, alpha=0.3)

    # 3. Returns Distribution
    plt.subplot(2, 2, 3)
    df["returns"].hist(bins=50, alpha=0.7, color="green")
    plt.axvline(
        df["returns"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {df['returns'].mean():.3f}%",
    )
    plt.title(f"{pair_name} {timeframe} - Returns Distribution")
    plt.xlabel("Returns (%)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 4. RSI Chart
    plt.subplot(2, 2, 4)
    if "rsi" in df.columns:
        plt.plot(df.index, df["rsi"], color="purple", alpha=0.8)
        plt.axhline(70, color="red", linestyle="--", alpha=0.5, label="Overbought (70)")
        plt.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold (30)")
        plt.title(f"{pair_name} {timeframe} - RSI")
        plt.xlabel("Time")
        plt.ylabel("RSI")
        plt.ylim(0, 100)
        plt.legend()
    else:
        plt.plot(df.index, df["price_change_pct"], alpha=0.7)
        plt.title(f"{pair_name} {timeframe} - Price Change %")
        plt.xlabel("Time")
        plt.ylabel("Price Change (%)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"{pair_name}_{timeframe}_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def analyze_trading_sessions(df: pd.DataFrame) -> dict:
    """Analyze performance by different trading sessions."""
    df = df.copy()
    df["hour"] = df["open_time"].dt.hour
    df["day_of_week"] = df["open_time"].dt.day_name()

    session_analysis = {}

    # Hourly analysis
    hourly_stats = (
        df.groupby("hour")
        .agg({"returns": ["mean", "std", "count"], "hl_spread_pct": "mean", "close": "mean"})
        .round(4)
    )
    session_analysis["hourly"] = hourly_stats

    # Daily analysis
    daily_stats = (
        df.groupby("day_of_week")
        .agg({"returns": ["mean", "std", "count"], "hl_spread_pct": "mean", "close": "mean"})
        .round(4)
    )
    session_analysis["daily"] = daily_stats

    # Find best and worst performing hours
    hourly_returns = df.groupby("hour")["returns"].mean()
    session_analysis["best_hour"] = hourly_returns.idxmax()
    session_analysis["worst_hour"] = hourly_returns.idxmin()

    return session_analysis


def main():
    """Main analysis function."""
    # Example usage - modify these paths to match your downloaded data
    data_dir = Path(".")

    # Look for data files
    json_files = list(data_dir.glob("*.json"))
    parquet_files = list(data_dir.glob("*.parquet"))

    if not json_files and not parquet_files:
        print("No data files found. Please run download_data.py first.")
        print("Looking for files matching patterns: *.json, *.parquet")
        return

    # Analyze first available file
    if json_files:
        filepath = json_files[0]
        df = load_json_data(filepath)
        print(f"Analyzing JSON file: {filepath}")
    else:
        filepath = parquet_files[0]
        df = load_parquet_data(filepath)
        print(f"Analyzing Parquet file: {filepath}")

    # Extract pair and timeframe from filename
    filename = filepath.stem
    parts = filename.split("_")
    pair_name = parts[0] if parts else "Unknown"
    timeframe = parts[1] if len(parts) > 1 else "Unknown"

    print(f"\n=== Analysis for {pair_name} {timeframe} ===")

    # Basic statistics
    print("\n1. Basic Statistics:")
    stats = calculate_basic_stats(df)
    print(f"Total candles: {stats['total_candles']:,}")
    print(f"Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
    print(
        f"Price range: ${stats['price_stats']['min_price']:,.2f} - ${stats['price_stats']['max_price']:,.2f}"
    )
    print(f"Average close price: ${stats['price_stats']['avg_close']:,.2f}")
    print(f"Price volatility (std): ${stats['price_stats']['price_volatility']:,.2f}")

    # Add technical indicators
    print("\n2. Adding Technical Indicators...")
    df = add_technical_indicators(df)

    # Pattern analysis
    print("\n3. Price Pattern Analysis:")
    patterns = analyze_price_patterns(df)
    print(f"Bullish candles: {patterns['bullish_candles']:,} ({patterns['bullish_ratio']:.1%})")
    print(f"Bearish candles: {patterns['bearish_candles']:,}")
    print(f"Doji patterns: {patterns['doji_candles']:,} ({patterns['doji_ratio']:.1%})")
    print(f"Hammer patterns: {patterns['hammer_candles']:,}")
    print(f"Average body size: ${patterns['avg_body_size']:.2f}")

    # Trading session analysis
    print("\n4. Trading Session Analysis:")
    try:
        session_stats = analyze_trading_sessions(df)
        print(f"Best performing hour: {session_stats['best_hour']}:00")
        print(f"Worst performing hour: {session_stats['worst_hour']}:00")
    except Exception as e:
        print(f"Session analysis failed: {e}")

    # Technical indicator summary
    print("\n5. Technical Indicators Summary:")
    if "rsi" in df.columns:
        current_rsi = df["rsi"].iloc[-1]
        print(f"Current RSI: {current_rsi:.1f}")
        if current_rsi > 70:
            print("  → Potentially overbought")
        elif current_rsi < 30:
            print("  → Potentially oversold")
        else:
            print("  → Neutral territory")

    if "macd" in df.columns:
        current_macd = df["macd"].iloc[-1]
        current_signal = df["macd_signal"].iloc[-1]
        print(f"MACD: {current_macd:.2f}, Signal: {current_signal:.2f}")
        if current_macd > current_signal:
            print("  → Bullish momentum")
        else:
            print("  → Bearish momentum")

    # Create visualizations
    print("\n6. Creating Visualizations...")
    try:
        create_visualizations(df, pair_name, timeframe, Path("./analysis_plots"))
        print("Visualizations saved to ./analysis_plots/")
    except Exception as e:
        print(f"Visualization creation failed: {e}")
        print(
            "This might be due to missing matplotlib/seaborn. Install with: pip install matplotlib seaborn"
        )

    # Save processed data with indicators
    output_file = f"{filename}_with_indicators.parquet"
    try:
        df.to_parquet(output_file, index=False)
        print(f"\nProcessed data with indicators saved to: {output_file}")
    except Exception as e:
        print(f"Failed to save processed data: {e}")

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
