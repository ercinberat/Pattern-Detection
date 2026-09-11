"""
Main entry point: runs the pipeline as it exists today for a single
ticker - fetch daily price history (Stage 1), find its swing pivots
(Stage 2), detect triangle/bull-flag candidates (Stage 3), compute
confirmation indicator features for each one (Stage 4), optionally label
each pattern's outcome (Stage 5), and plot the result (Stage 5b) - so the
pieces built so far can be exercised end-to-end without importing each
module by hand.

As later stages (modeling, backtesting) are built, this script is where
they get wired into the same end-to-end run.
"""

import argparse
import functools

from scripts.fetch_real_data import fetch_daily_price_history
from src.charting import plot_chart
from src.indicators import (
    INDICATOR_COMBINATIONS,
    compute_adx_dmi,
    compute_atr,
    compute_bollinger_bands,
    compute_donchian_channel,
    compute_macd,
    compute_obv,
    compute_relative_strength,
    compute_rsi,
)
from src.labeling import label_patterns
from src.patterns import (
    BullFlagPattern,
    TrianglePattern,
    deduplicate_bull_flags,
    deduplicate_triangles,
    detect_bull_flags,
    detect_triangles,
    find_pivots,
)

# Combination #5 needs a second ticker's data to compare against; SPY (an
# S&P 500 ETF) is used as a general-market benchmark by default.
_DEFAULT_BENCHMARK_TICKER = "SPY"


def run(
    ticker: str,
    pivot_order: int = 5,
    period: str = "2y",
    indicator_number: int = None,
    label_outcomes: bool = False,
    save_path: str = None,
) -> None:
    """
    Fetch a ticker's daily price history, find its swing pivots, and
    detect triangle/bull-flag candidates. If indicator_number is given,
    also compute confirmation indicator features for each detected
    pattern using that Stage 4 indicator combination. If label_outcomes
    is True, also label each detected pattern's outcome (Stage 5). Then
    plot the result.

    indicator_number: which entry of indicators.INDICATOR_COMBINATIONS to
        run, numbered to match PLAN.md's Stage 4 table (e.g. 1 = Bollinger
        Band squeeze + volume surge). If None (the default), Stage 4 is
        skipped entirely - only Stages 1-3 and the chart run.
    label_outcomes: if True, run src.labeling.label_patterns() on every
        detected pattern using its default fixed target/stop/time exit
        rule, print a win-rate summary, and draw each labeled trade on
        the chart.
    save_path: passed straight through to plot_chart() - if given, saves
        the chart to this file instead of opening a browser tab. Useful
        for scripted/repeated runs, e.g. scripts/end_to_end_test.py.
    """
    price_data = fetch_daily_price_history(ticker, period=period)
    pivots = find_pivots(price_data, order=pivot_order)

    # detect_triangles()/detect_bull_flags() each find every overlapping
    # window candidate; collapse those down to one per cluster so the
    # chart doesn't turn into a hairball of near-duplicate trendlines/flags.
    triangles = deduplicate_triangles(detect_triangles(pivots))
    bull_flags = deduplicate_bull_flags(detect_bull_flags(price_data))
    patterns = triangles + bull_flags

    price_overlays = None
    extra_panels = None

    if indicator_number is not None:
        combination = INDICATOR_COMBINATIONS[indicator_number]

        # Combination #5 additionally needs a benchmark ticker's data -
        # fetch it and bind it in, so it can be called the same
        # (price_data, pattern) way as every other combination below.
        compute_features = combination["compute_features"]
        if combination.get("needs_benchmark"):
            benchmark_data = fetch_daily_price_history(_DEFAULT_BENCHMARK_TICKER, period=period)
            compute_features = functools.partial(compute_features, benchmark_data=benchmark_data)

        # Stage 4: for every detected pattern, compute the chosen
        # indicator combination's features - printed to the console since
        # these are per-pattern confirmation scores, not a chart shape of
        # their own.
        print(f"Indicator #{indicator_number}: {combination['name']}")
        for pattern in patterns:
            features = compute_features(price_data, pattern)
            if isinstance(pattern, TrianglePattern):
                print(f"  {pattern.triangle_type} triangle ending {pattern.end_date.date()}: {features}")
            elif isinstance(pattern, BullFlagPattern):
                print(f"  bull flag ending {pattern.flag_end_date.date()}: {features}")

        # Each combination's chart wiring differs by what kind of series
        # it produces: combinations #1, #3, and #5's price-scale series
        # (Bollinger Bands, Donchian Channel, 52-week high) go in
        # price_overlays; combination #2's ADX/DMI and MACD, #3's OBV,
        # #4's RSI and ATR, and #5's relative strength are on their own
        # unrelated scales and need a separate stacked panel each.
        if indicator_number == 1:
            bollinger_bands = compute_bollinger_bands(price_data)
            price_overlays = [
                {
                    "lines": {"Upper Band": bollinger_bands["upper_band"], "Lower Band": bollinger_bands["lower_band"]},
                    "colors": {"Upper Band": "#787b86", "Lower Band": "#787b86"},
                },
            ]
        elif indicator_number == 2:
            adx_dmi = compute_adx_dmi(price_data)
            macd = compute_macd(price_data)
            extra_panels = [
                {
                    "ylabel": "ADX / DMI",
                    "lines": {"ADX": adx_dmi["adx"], "+DI": adx_dmi["plus_di"], "-DI": adx_dmi["minus_di"]},
                    "colors": {"ADX": "#d1d4dc", "+DI": "#26a69a", "-DI": "#ef5350"},
                },
                {
                    "ylabel": "MACD",
                    "lines": {"MACD": macd["macd_line"], "Signal": macd["signal_line"]},
                    "bars": {"Histogram": macd["histogram"]},
                    "colors": {"MACD": "#42a5f5", "Signal": "#ffa726", "Histogram": "#787b86"},
                },
            ]
        elif indicator_number == 3:
            donchian = compute_donchian_channel(price_data)
            obv = compute_obv(price_data)
            price_overlays = [
                {
                    "lines": {"Donchian Upper": donchian["upper_channel"], "Donchian Lower": donchian["lower_channel"]},
                    "colors": {"Donchian Upper": "#26a69a", "Donchian Lower": "#ef5350"},
                },
            ]
            extra_panels = [
                {"ylabel": "OBV", "lines": {"OBV": obv}, "colors": {"OBV": "#42a5f5"}},
            ]
        elif indicator_number == 4:
            rsi = compute_rsi(price_data)
            atr = compute_atr(price_data)
            extra_panels = [
                {"ylabel": "RSI", "lines": {"RSI": rsi}, "colors": {"RSI": "#42a5f5"}},
                {"ylabel": "ATR", "lines": {"ATR": atr}, "colors": {"ATR": "#ffa726"}},
            ]
        elif indicator_number == 5:
            # The rolling 52-week high is computed directly here (rather
            # than via a dedicated indicators.py function) since it's a
            # one-line rolling max - compute_pct_from_52_week_high()
            # already does this internally, but returns the % distance
            # from it rather than the price level a chart overlay needs.
            # benchmark_data was already fetched above (needed for
            # compute_features too), reused here rather than fetched again.
            rolling_52_week_high = price_data["Close"].rolling(window=252, min_periods=1).max()
            relative_strength = compute_relative_strength(price_data, benchmark_data)
            price_overlays = [
                {"lines": {"52-Week High": rolling_52_week_high}, "colors": {"52-Week High": "#787b86"}},
            ]
            extra_panels = [
                {"ylabel": "Relative Strength", "lines": {"Relative Strength": relative_strength}, "colors": {"Relative Strength": "#ab47bc"}},
            ]

    labels = None
    if label_outcomes:
        # Stage 5: label whether each pattern's breakout actually
        # followed through, using the default fixed target/stop/time
        # exit rule (see src/labeling.py). Patterns too close to the end
        # of price_data to have a full holding period yet are silently
        # skipped by label_patterns() rather than counted as failures.
        labels = label_patterns(price_data, patterns)
        successful_count = sum(label["is_successful"] for label in labels)
        # Win rate (exit_reason == "target") is a strict, binary measure -
        # a pattern that times out at +7% counts the same as one that gets
        # stopped out at -5%, even though its return says otherwise. Mean
        # return_pct is reported alongside it so a near-miss like that
        # isn't invisible in the summary.
        mean_return_pct = sum(label["return_pct"] for label in labels) / len(labels) if labels else 0.0
        print(
            f"Labeled {len(labels)} of {len(patterns)} patterns "
            f"({successful_count} successful, mean return {mean_return_pct:.1f}%):"
        )
        for label in labels:
            pattern_type = "triangle" if isinstance(label["pattern"], TrianglePattern) else "bull flag"
            print(
                f"  {pattern_type} entered {label['entry_date'].date()}: "
                f"{label['exit_reason']} exit, return {label['return_pct']:.1f}%, successful={label['is_successful']}"
            )

    plot_chart(
        price_data,
        ticker=ticker,
        pivots=pivots,
        patterns=patterns,
        price_overlays=price_overlays,
        extra_panels=extra_panels,
        labels=labels,
        save_path=save_path,
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
    parser.add_argument(
        "--label",
        action="store_true",
        help="Label each detected pattern's outcome (Stage 5's fixed target/stop/time exit rule) and draw it on the chart",
    )
    args = parser.parse_args()

    run(args.ticker, pivot_order=args.order, period=args.period, indicator_number=args.indicator, label_outcomes=args.label)
