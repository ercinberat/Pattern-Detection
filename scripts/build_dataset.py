"""
Builds Stage 6's training dataset: runs the full detection + labeling
pipeline (fetch -> pivots -> triangle/bull-flag detection -> labeling)
across a universe of tickers, rather than just one, computes every Stage
4 indicator combination's features for each pattern, and saves the whole
thing - pattern, label, and features together - to a single file.

This is the multi-ticker data pipeline that PLAN.md's Stage 1 ("needs...
multi-ticker batch support") and Stage 3 ("Next" note) both call for:
Stage 6's model, and Stage 3's own threshold-validation sweep, both need
many more labeled examples than one ticker's history can provide before
either means anything statistically.
"""

import argparse
import functools
import os

import pandas as pd

from scripts.fetch_real_data import fetch_daily_price_history, fetch_sp500_tickers
from src.indicators import INDICATOR_COMBINATIONS
from src.labeling import label_patterns
from src.patterns import deduplicate_bull_flags, deduplicate_triangles, detect_bull_flags, detect_triangles, find_pivots

# Combination #5 needs a second ticker's data to compare against; SPY (an
# S&P 500 ETF) is used as a general-market benchmark, same as main.py.
_DEFAULT_BENCHMARK_TICKER = "SPY"


def build_labeled_dataset(
    tickers: list,
    period: str = "2y",
    pivot_order: int = 5,
    benchmark_ticker: str = _DEFAULT_BENCHMARK_TICKER,
) -> pd.DataFrame:
    """
    Run pivot/pattern detection, labeling, and every Stage 4 indicator
    combination for every ticker in `tickers`, and combine the results
    into one DataFrame - one row per labeled pattern, features and all.

    Tickers that fail to fetch (delisted, too new to have `period` of
    history, a typo, etc.) are skipped with a printed message rather than
    stopping the whole run - a handful of bad tickers out of hundreds
    shouldn't block building the dataset from all the good ones.

    Returns a DataFrame with one row per labeled pattern:
        ticker, pattern_type, entry_date, entry_price, exit_date,
        exit_price, exit_reason, return_pct, is_successful, plus every
        indicator combination's feature columns, prefixed with
        "indicator{N}_" (e.g. combination #1's "is_squeezed" becomes
        "indicator1_is_squeezed") so all five combinations' columns can
        sit side by side without colliding.
    (label_patterns()'s "pattern" object itself is dropped after its
    features are computed, since a TrianglePattern/BullFlagPattern
    doesn't serialize cleanly to a flat file.)
    """
    benchmark_data = fetch_daily_price_history(benchmark_ticker, period=period)
    all_rows = []

    for ticker_index, ticker in enumerate(tickers, start=1):
        try:
            price_data = fetch_daily_price_history(ticker, period=period)
            pivots = find_pivots(price_data, order=pivot_order)
            triangles = deduplicate_triangles(detect_triangles(pivots))
            bull_flags = deduplicate_bull_flags(detect_bull_flags(price_data))
            labels = label_patterns(price_data, triangles + bull_flags)

            for label in labels:
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
                }

                for indicator_number, combination in INDICATOR_COMBINATIONS.items():
                    compute_features = combination["compute_features"]
                    if combination.get("needs_benchmark"):
                        compute_features = functools.partial(compute_features, benchmark_data=benchmark_data)
                    features = compute_features(price_data, label["pattern"])
                    for feature_name, feature_value in features.items():
                        row[f"indicator{indicator_number}_{feature_name}"] = feature_value

                all_rows.append(row)
        except Exception as error:
            print(f"  [{ticker_index}/{len(tickers)}] {ticker}: skipped ({error})")
            continue

        print(f"  [{ticker_index}/{len(tickers)}] {ticker}: {len(labels)} labeled patterns")

    return pd.DataFrame(all_rows)


def print_summary(dataset: pd.DataFrame) -> None:
    """
    Print an overall win-rate summary plus a per-ticker breakdown, so
    results can be sanity-checked both in aggregate and stock-by-stock -
    e.g. to catch one ticker behaving very differently from the rest.

    Win rate (the fraction of patterns whose exit_reason is "target") is a
    strict, binary measure - a pattern that times out at +7% counts the
    same as one that gets stopped out at -5%, even though its actual
    return says otherwise. Mean return_pct is reported alongside win rate
    everywhere below so that kind of near-miss isn't invisible.
    """
    if not len(dataset):
        print("No patterns found.")
        return

    print(
        f"Overall: {len(dataset)} patterns, win rate {dataset['is_successful'].mean():.1%}, "
        f"mean return {dataset['return_pct'].mean():.1f}%"
    )
    print(dataset["exit_reason"].value_counts().to_string())

    per_ticker = dataset.groupby("ticker").agg(
        patterns=("is_successful", "size"),
        wins=("is_successful", "sum"),
        mean_return_pct=("return_pct", "mean"),
    )
    per_ticker["win_rate"] = per_ticker["wins"] / per_ticker["patterns"]
    per_ticker = per_ticker.sort_values("win_rate", ascending=False)

    print(f"\nPer ticker ({len(per_ticker)} tickers with at least one pattern):")
    print(per_ticker.to_string(formatters={"win_rate": "{:.0%}".format, "mean_return_pct": "{:.1f}%".format}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Stage 6's labeled training dataset across a universe of tickers.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only scan the first N tickers of the universe, for a quick test run (default: scan all of them).",
    )
    parser.add_argument("--period", default="2y", help="How much daily history to fetch per ticker (default: 2y).")
    parser.add_argument(
        "--output",
        default="data/labeled_patterns.csv",
        help="Where to save the combined dataset (default: data/labeled_patterns.csv).",
    )
    args = parser.parse_args()

    universe = fetch_sp500_tickers()
    if args.limit:
        universe = universe[: args.limit]

    print(f"Scanning {len(universe)} tickers...")
    dataset = build_labeled_dataset(universe, period=args.period)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    dataset.to_csv(args.output, index=False)

    print(f"\nSaved {len(dataset)} labeled patterns to {args.output}\n")
    print_summary(dataset)
