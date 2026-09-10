"""
Downloads daily OHLCV (open/high/low/close/volume) price history for a
stock ticker using the yfinance library, which pulls data from Yahoo
Finance for free.

This is the first stage of the pipeline described in PLAN.md: before we can
detect any patterns, we need a clean daily price history to detect them on.
"""

import pandas as pd
import yfinance as yf


def fetch_daily_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for a single stock ticker.

    ticker: the stock's ticker symbol, e.g. "AAPL".
    period: how far back to pull data, using yfinance's period strings
        (e.g. "1y", "2y", "5y", "max"). Two years is enough history to see
        several triangle/flag setups while staying fast to download.

    Returns a DataFrame indexed by date, with columns:
        Open, High, Low, Close, Volume
    """
    # yf.Ticker(...).history(...) returns one row per trading day, already
    # sorted oldest to newest, which is what the pivot/pattern detection
    # stages downstream will expect.
    price_history = yf.Ticker(ticker).history(period=period, interval="1d")

    if price_history.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'. Check that the symbol is correct.")

    # Yahoo Finance includes dividend/stock-split columns we don't need for
    # pattern detection, so keep just the price/volume columns.
    price_history = price_history[["Open", "High", "Low", "Close", "Volume"]]

    return price_history


if __name__ == "__main__":
    aapl_daily_data = fetch_daily_price_history("AAPL")
    print(aapl_daily_data)
