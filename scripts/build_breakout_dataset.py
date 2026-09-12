"""
Builds a second version of Stage 6's training dataset: indicator features
read at each pattern's real breakout day (found via find_breakout_date(),
src/patterns.py) instead of at its detection window's end_date/flag_end_date.

Why this is a separate script rather than a change to build_dataset.py:
the label itself - entry_date, return_pct, is_successful, all of Stage
5's fixed target/stop/time exit rule - is left completely unchanged, so
data/labeled_patterns.csv stays a valid dataset in its own right ("what
happened to this trade"). This script asks a different question about
the exact same trades: "what did the indicators actually look like right
as the real move started", which "Right Signal, Wrong Day" (see
PLAN.md's Stage 6 notes) showed gives a meaningfully different - and
more useful - answer than reading them off a detection window's often-
stale end date.

Patterns whose geometry never actually got broken within
find_breakout_date()'s search window (about 5% of all patterns, in the
full-universe run this was built against) are dropped entirely here,
rather than falling back to end_date for just those - mixing two
different evaluation rules into one dataset would reintroduce the exact
timing problem this script exists to avoid.
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

# Combination #5 needs a second ticker's data to compare against; SPY (an
# S&P 500 ETF) is used as a general-market benchmark, same as
# build_dataset.py and main.py.
_DEFAULT_BENCHMARK_TICKER = "SPY"


def build_breakout_dataset(
    tickers: list,
    period: str = "2y",
    pivot_order: int = 5,
    benchmark_ticker: str = _DEFAULT_BENCHMARK_TICKER,
    max_search_days: int = 20,
) -> pd.DataFrame:
    """
    Run pivot/pattern detection and labeling for every ticker in
    `tickers`, exactly like build_labeled_dataset() does - but for every
    labeled pattern, find its real breakout day and compute every Stage 4
    indicator combination's features there instead of at the pattern's
    own end_date/flag_end_date.

    Returns a DataFrame with one row per pattern that had a real breakout
    found (patterns that never broke out within max_search_days are
    dropped - see this file's module docstring for why):
        ticker, pattern_type, entry_date, entry_price, exit_date,
        exit_price, exit_reason, return_pct, is_successful (all from the
        pattern's existing label, unchanged), end_date (the detection
        window's own evaluation date, kept for reference/comparison),
        breakout_date, breakout_direction, lag_days (calendar days
        between end_date and breakout_date), plus every indicator
        combination's feature columns, prefixed "indicator{N}_" exactly
        like build_labeled_dataset() - but computed at breakout_date.
    """
    benchmark_data = fetch_daily_price_history(benchmark_ticker, period=period)
    all_rows = []
    dropped_no_breakout = 0

    for ticker_index, ticker in enumerate(tickers, start=1):
        try:
            price_data = fetch_daily_price_history(ticker, period=period)
            pivots = find_pivots(price_data, order=pivot_order)
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

        kept_this_ticker = 0
        for label in labels:
            pattern = label["pattern"]

            breakout = find_breakout_date(price_data, pattern, max_search_days=max_search_days)
            if breakout is None:
                dropped_no_breakout += 1
                continue

            end_date = pattern_evaluation_date(pattern)
            pattern_at_breakout = pattern_as_of_breakout(pattern, breakout["breakout_date"])

            row = {
                "ticker": ticker,
                "pattern_type": label["pattern_type"],
                "entry_date": label["entry_date"],
                "entry_price": label["entry_price"],
                "exit_date": label["exit_date"],
                "exit_price": label["exit_price"],
                "exit_reason": label["exit_reason"],
                "return_pct": label["return_pct"],
                "is_successful": label["is_successful"],
                "end_date": end_date,
                "breakout_date": breakout["breakout_date"],
                "breakout_direction": breakout["direction"],
                "lag_days": (breakout["breakout_date"] - end_date).days,
            }

            for indicator_number, combination in INDICATOR_COMBINATIONS.items():
                compute_features = combination["compute_features"]
                if combination.get("needs_benchmark"):
                    compute_features = functools.partial(compute_features, benchmark_data=benchmark_data)
                # The only change from build_labeled_dataset(): features
                # are computed against pattern_at_breakout, so
                # pattern_evaluation_date() inside compute_features
                # resolves to the real breakout day instead of end_date.
                features = compute_features(price_data, pattern_at_breakout)
                for feature_name, feature_value in features.items():
                    row[f"indicator{indicator_number}_{feature_name}"] = feature_value

            all_rows.append(row)
            kept_this_ticker += 1

        print(f"  [{ticker_index}/{len(tickers)}] {ticker}: {kept_this_ticker} patterns kept (breakout found)")

    dataset = pd.DataFrame(all_rows)
    print(
        f"\n{len(dataset)} patterns kept, {dropped_no_breakout} dropped "
        f"(no breakout found within {max_search_days} days)."
    )
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build Stage 6's training dataset with indicator features read at each pattern's real breakout day."
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
        help="How many bars past end_date to keep looking for a real breakout before dropping the pattern (default: 20).",
    )
    parser.add_argument(
        "--output",
        default="data/breakout_labeled_patterns.csv",
        help="Where to save the dataset (default: data/breakout_labeled_patterns.csv).",
    )
    args = parser.parse_args()

    universe = fetch_sp500_tickers()
    if args.limit:
        universe = universe[: args.limit]

    print(f"Scanning {len(universe)} tickers...")
    dataset = build_breakout_dataset(universe, period=args.period, max_search_days=args.max_search_days)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(f"\nSaved {len(dataset)} rows to {args.output}")
