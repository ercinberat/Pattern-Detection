"""
Measures how much difference it makes to read Stage 4's indicators at a
pattern's own end_date/flag_end_date - the last bar of its detection
window - versus at the real breakout/breakdown candle found by
find_breakout_date() (src/patterns.py).

Why this exists: a real case (EXPE's descending triangle, documented in
PLAN.md's Stage 3 notes) showed those two dates can land days apart -
end_date landed 3 trading days after a +17.6% earnings gap, because the
detection window that survived deduplicate_triangles()'s tie-break still
contained the actual gap day. "The Confirmation Paradox" (more confirming
signals correlating with WORSE outcomes, also in PLAN.md's Stage 6 notes)
raised the possibility that at least part of that result comes from
reading the indicators too late relative to the real move, rather than
confirmation itself being a bad sign. This script tests that directly.

It's built as a reusable measure, not a one-off check on today's five
indicator combinations: it reads whichever combinations are registered in
src/indicators.py's INDICATOR_COMBINATIONS, so any indicator added there
in the future gets the same before/after comparison automatically, aside
from the one thing that does need updating by hand - see
_BOOLEAN_FEATURE_NAMES below.
"""

import argparse
import functools
import os

import pandas as pd

from scripts.fetch_real_data import fetch_daily_price_history, fetch_sp500_tickers
from src.indicators import INDICATOR_COMBINATIONS
from src.labeling import label_patterns
from src.patterns import (
    deduplicate_bull_flags,
    deduplicate_triangles,
    detect_bull_flags,
    detect_triangles,
    find_breakout_date,
    find_pivots,
    pattern_as_of_breakout,
    pattern_evaluation_date,
    remove_bull_flags_inside_wedges,
)

_DEFAULT_BENCHMARK_TICKER = "SPY"

# The 12 boolean confirmation flags across all 5 indicator combinations -
# the same list "The Confirmation Paradox" artifact used to build
# confirmation_count. Unlike INDICATOR_COMBINATIONS itself, this list
# isn't derived automatically (a combination's feature dict can also hold
# non-boolean values like raw indicator readings), so adding a 6th
# combination later means adding its boolean feature names here too.
_BOOLEAN_FEATURE_NAMES = {
    1: ["is_squeezed", "is_volume_surge"],
    2: ["is_trending", "is_bullish_direction", "is_macd_bullish_flip"],
    3: ["is_donchian_breakout", "is_obv_new_high"],
    4: ["was_basing", "is_momentum_shift", "is_atr_expanding"],
    5: ["is_near_52_week_high", "is_outperforming_benchmark"],
}


def _count_confirming_flags(price_data: pd.DataFrame, benchmark_data: pd.DataFrame, pattern) -> int:
    """
    Run every registered indicator combination against `pattern`,
    evaluated at whatever date pattern_evaluation_date(pattern) currently
    points to, and count how many of the 12 boolean flags came back True.
    """
    confirming_flags = 0
    for indicator_number, combination in INDICATOR_COMBINATIONS.items():
        compute_features = combination["compute_features"]
        if combination.get("needs_benchmark"):
            compute_features = functools.partial(compute_features, benchmark_data=benchmark_data)
        features = compute_features(price_data, pattern)
        for feature_name in _BOOLEAN_FEATURE_NAMES[indicator_number]:
            if features[feature_name]:
                confirming_flags += 1
    return confirming_flags


def measure_indicator_lag(
    tickers: list,
    period: str = "2y",
    benchmark_ticker: str = _DEFAULT_BENCHMARK_TICKER,
    max_search_days: int = 20,
) -> pd.DataFrame:
    """
    For every labeled pattern across `tickers`, find the real
    breakout/breakdown day and recompute every indicator combination's
    confirmation flags both at the pattern's own end_date/flag_end_date
    and at that real breakout day, so the two readings can be compared.

    The pattern's existing label (return_pct, is_successful) is carried
    over unchanged - this only asks what the indicators would have shown
    at a different evaluation date, not whether entering/exiting
    differently would have helped.

    max_search_days: passed through to find_breakout_date() - patterns
    whose geometry never gets broken within this many days (they just
    fizzle sideways) have no "at breakout" date to compare against, so
    they're dropped from the returned rows, but counted and reported by
    print_summary() below.

    Returns a DataFrame with one row per pattern that had a breakout
    found: ticker, pattern_type, lag_days (calendar days between end_date
    and the real breakout), breakout_direction, return_pct, is_successful,
    confirmation_count_at_end_date, confirmation_count_at_breakout.
    """
    benchmark_data = fetch_daily_price_history(benchmark_ticker, period=period)
    rows = []
    total_patterns = 0
    patterns_with_no_breakout_found = 0

    for ticker_index, ticker in enumerate(tickers, start=1):
        try:
            price_data = fetch_daily_price_history(ticker, period=period)
            pivots = find_pivots(price_data)
            triangles = deduplicate_triangles(detect_triangles(pivots))
            bull_flags = deduplicate_bull_flags(detect_bull_flags(price_data))
            # A bull flag fully inside a wedge's date range is just a
            # smaller piece of the same move the wedge already describes,
            # not an independent setup.
            bull_flags = remove_bull_flags_inside_wedges(triangles, bull_flags)
            labels = label_patterns(price_data, triangles + bull_flags)
        except Exception as error:
            print(f"  [{ticker_index}/{len(tickers)}] {ticker}: skipped ({error})")
            continue

        for label in labels:
            total_patterns += 1
            pattern = label["pattern"]

            breakout = find_breakout_date(price_data, pattern, max_search_days=max_search_days)
            if breakout is None:
                patterns_with_no_breakout_found += 1
                continue

            end_date = pattern_evaluation_date(pattern)
            lag_days = (breakout["breakout_date"] - end_date).days
            pattern_at_breakout = pattern_as_of_breakout(pattern, breakout["breakout_date"])

            rows.append(
                {
                    "ticker": ticker,
                    "pattern_type": label["pattern_type"],
                    "lag_days": lag_days,
                    "breakout_direction": breakout["direction"],
                    "return_pct": label["return_pct"],
                    "is_successful": label["is_successful"],
                    "confirmation_count_at_end_date": _count_confirming_flags(price_data, benchmark_data, pattern),
                    "confirmation_count_at_breakout": _count_confirming_flags(price_data, benchmark_data, pattern_at_breakout),
                }
            )

        print(f"  [{ticker_index}/{len(tickers)}] {ticker}: {len(labels)} labeled patterns")

    measurements = pd.DataFrame(rows)
    print(
        f"\n{len(measurements)} of {total_patterns} labeled patterns had a real breakout found within "
        f"{max_search_days} days of their own end date "
        f"({patterns_with_no_breakout_found} never broke out of their own lines/box in that window)."
    )
    return measurements


def print_summary(measurements: pd.DataFrame) -> None:
    """
    Print the comparison this script exists to make: does the
    confirmation_count-vs-return relationship look different when the
    indicators are read at the real breakout day instead of at the
    detection window's own end date?
    """
    if not len(measurements):
        print("No patterns with a found breakout to compare.")
        return

    print(f"\nLag between end_date and the real breakout (days): "
          f"mean {measurements['lag_days'].mean():.1f}, median {measurements['lag_days'].median():.0f}, "
          f"max {measurements['lag_days'].max():.0f}")

    triangle_directions = measurements[measurements["pattern_type"] == "triangle"]["breakout_direction"]
    if len(triangle_directions):
        print(
            f"Triangle breakout direction actually observed: "
            f"{(triangle_directions == 'up').mean():.0%} up, {(triangle_directions == 'down').mean():.0%} down "
            f"(a triangle's classified type - symmetrical/ascending/descending - doesn't fix its breakout direction)"
        )

    end_date_correlation = measurements["confirmation_count_at_end_date"].corr(measurements["return_pct"])
    breakout_correlation = measurements["confirmation_count_at_breakout"].corr(measurements["return_pct"])
    print(
        f"\nCorrelation of confirmation_count with return_pct:\n"
        f"  measured at end_date (current pipeline): {end_date_correlation:+.3f}\n"
        f"  measured at the real breakout day:        {breakout_correlation:+.3f}"
    )

    print("\nMean return by confirmation_count bucket, both timings side by side:")
    comparison = pd.DataFrame(
        {
            "at_end_date": measurements.groupby("confirmation_count_at_end_date")["return_pct"].mean(),
            "at_end_date_n": measurements.groupby("confirmation_count_at_end_date")["return_pct"].size(),
            "at_breakout": measurements.groupby("confirmation_count_at_breakout")["return_pct"].mean(),
            "at_breakout_n": measurements.groupby("confirmation_count_at_breakout")["return_pct"].size(),
        }
    )
    print(
        comparison.to_string(
            formatters={
                "at_end_date": "{:+.1f}%".format,
                "at_breakout": "{:+.1f}%".format,
            },
            na_rep="-",
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Stage 4 indicator readings at a pattern's end_date vs. at its real breakout day."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only scan the first N tickers of the S&P 500 universe, for a quick test run (default: scan all of them).",
    )
    parser.add_argument("--period", default="2y", help="How much daily history to fetch per ticker (default: 2y).")
    parser.add_argument(
        "--max-search-days",
        type=int,
        default=20,
        help="How many bars past end_date to keep looking for a real breakout before giving up (default: 20, matching Stage 5's max_holding_days).",
    )
    parser.add_argument(
        "--output",
        default="data/indicator_lag_measurements.csv",
        help="Where to save the per-pattern comparison (default: data/indicator_lag_measurements.csv).",
    )
    args = parser.parse_args()

    universe = fetch_sp500_tickers()
    if args.limit:
        universe = universe[: args.limit]

    print(f"Scanning {len(universe)} tickers...")
    measurements = measure_indicator_lag(universe, period=args.period, max_search_days=args.max_search_days)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    measurements.to_csv(args.output, index=False)
    print(f"\nSaved {len(measurements)} rows to {args.output}")

    print_summary(measurements)
