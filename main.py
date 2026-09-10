"""
Main entry point: runs the pipeline as it exists today for a single
ticker - fetch daily price history (Stage 1), find its swing pivots
(Stage 2), detect triangle/bull-flag candidates (Stage 3), and plot the
result (Stage 5b) - so the pieces built so far can be exercised end-to-end
without importing each module by hand.

As later stages (confirmation indicators, labeling, modeling,
backtesting) are built, this script is where they get wired into the same
end-to-end run.
"""

import argparse

from scripts.fetch_real_data import fetch_daily_price_history
from src.charting import plot_chart
from src.patterns import deduplicate_triangles, detect_bull_flags, detect_triangles, find_pivots


def run(ticker: str, pivot_order: int = 5, period: str = "2y") -> None:
    """
    Fetch a ticker's daily price history, find its swing pivots, detect
    triangle/bull-flag candidates, and plot the result.
    """
    price_data = fetch_daily_price_history(ticker, period=period)
    pivots = find_pivots(price_data, order=pivot_order)

    # detect_triangles() finds every overlapping window candidate; collapse
    # those down to one per cluster so the chart doesn't turn into a
    # hairball of near-duplicate trendlines.
    triangles = deduplicate_triangles(detect_triangles(pivots))
    bull_flags = detect_bull_flags(price_data)

    plot_chart(price_data, ticker=ticker, pivots=pivots, patterns=triangles + bull_flags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch daily price data, find swing pivots, and plot the chart for a stock ticker."
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default="AAPL",
        help="Stock ticker symbol, e.g. AAPL (default: AAPL)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=5,
        help="Pivot sensitivity - bigger values find fewer, more significant swings (default: 5)",
    )
    parser.add_argument(
        "--period",
        default="2y",
        help="How much history to fetch, as a yfinance period string like 1y/2y/5y/max (default: 2y)",
    )
    args = parser.parse_args()

    run(args.ticker, pivot_order=args.order, period=args.period)
