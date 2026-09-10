"""
Main entry point: runs the pipeline as it exists today for a single
ticker - fetch daily price history (Stage 1), find its swing pivots
(Stage 2), detect triangle/bull-flag candidates (Stage 3), compute
confirmation indicator features for each one (Stage 4), and plot the
result (Stage 5b) - so the pieces built so far can be exercised end-to-end
without importing each module by hand.

As the rest of Stage 4 and later stages (labeling, modeling, backtesting)
are built, this script is where they get wired into the same end-to-end
run.
"""

import argparse

from scripts.fetch_real_data import fetch_daily_price_history
from src.charting import plot_chart
from src.indicators import INDICATOR_COMBINATIONS, compute_bollinger_bands
from src.patterns import (
    BullFlagPattern,
    TrianglePattern,
    deduplicate_triangles,
    detect_bull_flags,
    detect_triangles,
    find_pivots,
)


def run(ticker: str, pivot_order: int = 5, period: str = "2y", indicator_number: int = None) -> None:
    """
    Fetch a ticker's daily price history, find its swing pivots, and
    detect triangle/bull-flag candidates. If indicator_number is given,
    also compute confirmation indicator features for each detected
    pattern using that Stage 4 indicator combination. Then plot the
    result.

    indicator_number: which entry of indicators.INDICATOR_COMBINATIONS to
        run, numbered to match PLAN.md's Stage 4 table (e.g. 1 = Bollinger
        Band squeeze + volume surge). If None (the default), Stage 4 is
        skipped entirely - only Stages 1-3 and the chart run.
    """
    price_data = fetch_daily_price_history(ticker, period=period)
    pivots = find_pivots(price_data, order=pivot_order)

    # detect_triangles() finds every overlapping window candidate; collapse
    # those down to one per cluster so the chart doesn't turn into a
    # hairball of near-duplicate trendlines.
    triangles = deduplicate_triangles(detect_triangles(pivots))
    bull_flags = detect_bull_flags(price_data)
    patterns = triangles + bull_flags

    bollinger_bands = None

    if indicator_number is not None:
        # Stage 4: for every detected pattern, compute the chosen
        # indicator combination's features - printed to the console since
        # these are per-pattern confirmation scores, not a chart shape of
        # their own.
        combination = INDICATOR_COMBINATIONS[indicator_number]
        print(f"Indicator #{indicator_number}: {combination['name']}")
        for pattern in patterns:
            features = combination["compute_features"](price_data, pattern)
            if isinstance(pattern, TrianglePattern):
                print(f"  {pattern.triangle_type} triangle ending {pattern.end_date.date()}: {features}")
            elif isinstance(pattern, BullFlagPattern):
                print(f"  bull flag ending {pattern.flag_end_date.date()}: {features}")

        # Combination #1's Bollinger Bands are drawn directly on the chart
        # so the squeeze can be seen alongside the pattern shapes.
        # Combinations #2-5 will need their own chart wiring added here
        # once they exist.
        if indicator_number == 1:
            bollinger_bands = compute_bollinger_bands(price_data)

    plot_chart(
        price_data,
        ticker=ticker,
        pivots=pivots,
        patterns=patterns,
        bollinger_bands=bollinger_bands,
    )


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
    parser.add_argument(
        "--indicator",
        type=int,
        default=None,
        choices=sorted(INDICATOR_COMBINATIONS.keys()),
        help="Which Stage 4 indicator combination to run, numbered per PLAN.md's table (default: none - Stage 4 is skipped)",
    )
    args = parser.parse_args()

    run(args.ticker, pivot_order=args.order, period=args.period, indicator_number=args.indicator)
