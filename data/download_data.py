#!/usr/bin/env python3

"""
CLI application to download candlestick data from Binance API.

Supports downloading OHLCV data for multiple cryptocurrency pairs with various
timeframes and saves data in JSON or Parquet format.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests


class BinanceDataDownloader:
    """Downloads candlestick data from Binance API."""

    BASE_URL = "https://api.binance.com/api/v3/klines"

    # Mapping of user-friendly pairs to Binance symbols
    PAIR_MAPPING = {"BTC/USDT": "BTCUSDT", "ETH/USDT": "ETHUSDT", "SOL/USDT": "SOLUSDT"}

    # Mapping of user-friendly timeframes to Binance intervals
    TIMEFRAME_MAPPING = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h"}

    def __init__(self):
        self.session = requests.Session()
        # Add headers to appear as a legitimate client
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; BinanceDataDownloader/1.0)"}
        )

    def date_to_timestamp(self, date_str: str) -> int:
        """Convert YYYYMMDD date string to millisecond timestamp."""
        try:
            date_obj = datetime.strptime(date_str, "%Y%m%d")
            # Set timezone to UTC for consistency with Binance
            date_obj = date_obj.replace(tzinfo=timezone.utc)
            return int(date_obj.timestamp() * 1000)
        except ValueError:
            raise ValueError(f"Invalid date format: {date_str}. Expected YYYYMMDD.")

    def fetch_klines(
        self, symbol: str, interval: str, start_time: int, limit: int = 1000
    ) -> List[List]:
        """Fetch kline data from Binance API."""
        params = {"symbol": symbol, "interval": interval, "startTime": start_time, "limit": limit}

        try:
            response = self.session.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from Binance API: {e}", file=sys.stderr)
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}", file=sys.stderr)
            return []

    def download_data(self, pair: str, timeframe: str, from_date: str) -> List[Dict[str, Any]]:
        """Download all candlestick data from the specified date to present."""
        symbol = self.PAIR_MAPPING[pair]
        interval = self.TIMEFRAME_MAPPING[timeframe]
        start_time = self.date_to_timestamp(from_date)

        print(f"Downloading {pair} {timeframe} data from {from_date}...")

        all_data = []
        batch_count = 0

        while True:
            # Fetch up to 1000 candles per request (Binance limit)
            klines = self.fetch_klines(symbol, interval, start_time)

            if not klines:
                break

            batch_count += 1
            batch_size = len(klines)

            # Convert raw kline data to structured format
            for kline in klines:
                candle_data = {
                    "open_time": int(kline[0]),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "close_time": int(kline[6]),
                }
                all_data.append(candle_data)

            print(f"  Batch {batch_count}: Downloaded {batch_size} candles")

            # If we got less than 1000 candles, we've reached the end
            if batch_size < 1000:
                break

            # Update start_time to the next millisecond after the last candle's close_time
            start_time = int(klines[-1][6]) + 1

            # Rate limiting: sleep briefly between requests to be respectful
            time.sleep(0.1)

        print(f"Total downloaded: {len(all_data)} candles")
        return all_data

    def save_data(
        self,
        data: List[Dict[str, Any]],
        pair: str,
        timeframe: str,
        from_date: str,
        format_type: str,
        output_dir: Path,
    ) -> None:
        """Save data to file in specified format."""
        if not data:
            print("No data to save.", file=sys.stderr)
            return

        # Create filename
        pair_clean = pair.replace("/", "")
        filename_base = f"{pair_clean}_{timeframe}_{from_date}"

        output_dir.mkdir(parents=True, exist_ok=True)

        if format_type == "json":
            output_file = output_dir / f"{filename_base}.json"
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Data saved to: {output_file}")

        elif format_type == "parquet":
            # Convert to DataFrame for parquet export
            df = pd.DataFrame(data)

            # Convert timestamp columns to datetime for better parquet support
            df["open_datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
            df["close_datetime"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

            output_file = output_dir / f"{filename_base}.parquet"
            df.to_parquet(output_file, index=False, engine="pyarrow")
            print(f"Data saved to: {output_file}")

        else:
            print(f"Unsupported format: {format_type}", file=sys.stderr)


def validate_arguments(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    # Validate timeframe
    valid_timeframes = list(BinanceDataDownloader.TIMEFRAME_MAPPING.keys())
    if args.timeframe not in valid_timeframes:
        print(
            f"Error: Invalid timeframe '{args.timeframe}'. "
            f"Supported: {', '.join(valid_timeframes)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate pairs
    valid_pairs = list(BinanceDataDownloader.PAIR_MAPPING.keys())
    invalid_pairs = [pair for pair in args.pairs if pair not in valid_pairs]
    if invalid_pairs:
        print(
            f"Error: Invalid pairs: {', '.join(invalid_pairs)}. "
            f"Supported: {', '.join(valid_pairs)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate date format
    try:
        datetime.strptime(args.from_date, "%Y%m%d")
    except ValueError:
        print(
            f"Error: Invalid date format '{args.from_date}'. "
            f"Expected YYYYMMDD format (e.g., 20250801).",
            file=sys.stderr,
        )
        sys.exit(1)

    # Validate format
    if args.format not in ["json", "parquet"]:
        print(f"Error: Invalid format '{args.format}'. Supported: json, parquet", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to handle CLI arguments and orchestrate data download."""
    parser = argparse.ArgumentParser(
        description="Download candlestick data from Binance API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download BTC/USDT 1-hour data from August 1, 2025 in parquet format
  python download_data.py --timeframe 1h --pairs BTC/USDT --from 20250801 --format parquet

  # Download multiple pairs with 5-minute timeframe in JSON format
  python download_data.py --timeframe 5m --pairs BTC/USDT ETH/USDT SOL/USDT --from 20250101 --format json
        """,  # noqa: E501
    )

    parser.add_argument(
        "--timeframe", required=True, help="Timeframe for candlestick data (1m, 5m, 15m, 30m, 1h)"
    )

    parser.add_argument(
        "--pairs",
        nargs="+",
        required=True,
        help="Trading pairs to download (BTC/USDT, ETH/USDT, SOL/USDT)",
    )

    parser.add_argument(
        "--from",
        dest="from_date",
        required=True,
        help="Start date in YYYYMMDD format (e.g., 20250801)",
    )

    parser.add_argument("--format", required=True, help="Output format (json, parquet)")

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for downloaded data (default: ./data)",
    )

    args = parser.parse_args()

    # Validate arguments
    validate_arguments(args)

    # Create downloader instance
    downloader = BinanceDataDownloader()

    # Download data for each requested pair
    for pair in args.pairs:
        try:
            data = downloader.download_data(pair, args.timeframe, args.from_date)
            downloader.save_data(
                data, pair, args.timeframe, args.from_date, args.format, args.output_dir
            )
            print()  # Add blank line between pairs for readability

        except Exception as e:
            print(f"Error downloading data for {pair}: {e}", file=sys.stderr)
            continue

    print("Download complete!")


if __name__ == "__main__":
    main()
