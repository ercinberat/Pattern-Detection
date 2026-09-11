"""
Downloads daily OHLCV (open/high/low/close/volume) price history for a
stock ticker using the yfinance library, which pulls data from Yahoo
Finance for free. Also fetches the list of tickers to scan (the S&P 500's
current constituents), for Stage 6's need to build a training set across
many tickers rather than just one.

This is the first stage of the pipeline described in PLAN.md: before we can
detect any patterns, we need a clean daily price history to detect them on.
"""

import io
import os

import pandas as pd
import requests
import yfinance as yf

# Where cached price data is stored, so re-running a script that fetches
# the same tickers doesn't re-download them from Yahoo Finance every
# time. Gitignored - this is local scratch data, not something to commit.
_CACHE_DIR = "data"


def fetch_daily_price_history(ticker: str, period: str = "2y", use_cache: bool = True) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars for a single stock ticker.

    ticker: the stock's ticker symbol, e.g. "AAPL".
    period: how far back to pull data, using yfinance's period strings
        (e.g. "1y", "2y", "5y", "max"). Two years is enough history to see
        several triangle/flag setups while staying fast to download.
    use_cache: if True (the default), save this ticker's data to a local
        CSV file under data/ the first time it's fetched, and reuse that
        file on later calls instead of re-downloading. Useful when
        iterating on a script that fetches the same tickers repeatedly -
        e.g. scanning hundreds of tickers for Stage 6's training data,
        where re-downloading everything on every run would be slow and
        put unnecessary load on Yahoo Finance. Set to False to always
        pull fresh data.

    Returns a DataFrame indexed by date, with columns:
        Open, High, Low, Close, Volume
    """
    cache_path = os.path.join(_CACHE_DIR, f"{ticker.replace('.', '_')}_{period}.csv")

    if use_cache and os.path.exists(cache_path):
        cached_data = pd.read_csv(cache_path, index_col=0)
        # read_csv's own parse_dates doesn't reliably parse the index back
        # into real Timestamps when the dates include a timezone offset
        # (e.g. "2024-09-12 00:00:00-04:00") - it silently leaves them as
        # plain strings instead, which breaks anything downstream that
        # calls date-like methods (e.g. .date()) on the index. Parsing
        # explicitly with pd.to_datetime() avoids that.
        cached_data.index = pd.to_datetime(cached_data.index, utc=True)
        cached_data.index.name = "Date"
        return cached_data

    # yf.Ticker(...).history(...) returns one row per trading day, already
    # sorted oldest to newest, which is what the pivot/pattern detection
    # stages downstream will expect.
    price_history = yf.Ticker(ticker).history(period=period, interval="1d")

    if price_history.empty:
        raise ValueError(f"No price data returned for ticker '{ticker}'. Check that the symbol is correct.")

    # Yahoo Finance includes dividend/stock-split columns we don't need for
    # pattern detection, so keep just the price/volume columns.
    price_history = price_history[["Open", "High", "Low", "Close", "Volume"]]

    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        price_history.to_csv(cache_path)

    return price_history


def fetch_sp500_tickers() -> list:
    """
    Fetch the current list of S&P 500 constituent ticker symbols, scraped
    from Wikipedia's "List of S&P 500 companies" page - a freely available,
    commonly used source for this that doesn't need an API key.

    This is a snapshot of today's index membership, not a point-in-time
    historical membership list - a stock added to or removed from the
    index over the last 2 years (the default fetch_daily_price_history
    window) won't necessarily have been a member for that whole period.
    That's a real limitation (a kind of survivorship bias) worth keeping
    in mind once this feeds into Stage 6's training data, but the index's
    current list is a reasonable, simple starting universe.

    Returns a list of ticker symbols as strings, e.g. ["AAPL", "MSFT", ...].
    """
    # Wikipedia blocks requests that don't look like they're coming from a
    # real browser, so a normal User-Agent header has to be set - the
    # default one urllib/pandas.read_html sends on its own gets a 403.
    wikipedia_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(wikipedia_url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()

    # Wrapped in StringIO rather than passed as a plain string, since
    # pandas/lxml would otherwise try to treat the string itself as a
    # filename or URL to open, rather than as HTML text to parse.
    constituent_tables = pd.read_html(io.StringIO(response.text))
    constituents = constituent_tables[0]

    # Wikipedia lists share classes with a dot (e.g. "BRK.B"), but Yahoo
    # Finance/yfinance expects a hyphen instead (e.g. "BRK-B").
    tickers = constituents["Symbol"].str.replace(".", "-", regex=False)

    return tickers.tolist()


if __name__ == "__main__":
    aapl_daily_data = fetch_daily_price_history("AAPL")
    print(aapl_daily_data)

    sp500_tickers = fetch_sp500_tickers()
    print(f"\nFetched {len(sp500_tickers)} S&P 500 tickers, first 10: {sp500_tickers[:10]}")
