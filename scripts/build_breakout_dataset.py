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
    TrianglePattern,
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


def _compute_market_regime_series(benchmark_data: pd.DataFrame) -> pd.DataFrame:
    """
    Two simple descriptors of the broader market's own condition, from
    the benchmark ticker's (SPY by default) own price series. Every
    other feature in this dataset describes one stock in isolation at
    one moment - nothing captures whether the market itself is trending
    or choppy right now, which plausibly affects whether an individual
    breakout follows through (see PLAN.md's Stage 6 notes).

    market_pct_from_50d_average: how far SPY's Close sits above (positive)
        or below (negative) its own 50-day moving average, as a % - a
        simple "is the market itself in an uptrend" measure. 50 days is a
        medium-term window, roughly 2.5 trading months.
    market_20d_volatility_pct: the standard deviation of SPY's daily %
        returns over the last 20 trading days - a simple realized-
        volatility measure. Higher means choppier, more uncertain recent
        conditions market-wide.
    """
    fifty_day_average = benchmark_data["Close"].rolling(window=50).mean()
    pct_from_50_day_average = (benchmark_data["Close"] - fifty_day_average) / fifty_day_average * 100

    daily_return_pct = benchmark_data["Close"].pct_change() * 100
    twenty_day_volatility = daily_return_pct.rolling(window=20).std()

    return pd.DataFrame(
        {
            "market_pct_from_50d_average": pct_from_50_day_average,
            "market_20d_volatility_pct": twenty_day_volatility,
        }
    )


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
        between end_date and breakout_date), pattern_high_r_squared/
        pattern_low_r_squared/pattern_contraction_pct (triangles) and
        pattern_pole_return_pct/pattern_flag_volume_ratio/
        pattern_retracement_pct (bull flags) - Stage 3's own geometric
        quality measures, 0 for whichever pair of fields doesn't apply to
        that row's pattern_type, market_pct_from_50d_average/
        market_20d_volatility_pct (see _compute_market_regime_series()),
        plus every indicator combination's feature columns, prefixed
        "indicator{N}_" exactly like build_labeled_dataset() - but
        computed at breakout_date.
    """
    benchmark_data = fetch_daily_price_history(benchmark_ticker, period=period)
    market_regime_series = _compute_market_regime_series(benchmark_data)
    all_rows = []
    dropped_no_breakout = 0

    for ticker_index, ticker in enumerate(tickers, start=1):
        try:
            price_data = fetch_daily_price_history(ticker, period=period)
            # The benchmark and this ticker don't always share the exact
            # same trading calendar (a holiday observed on one exchange
            # but not another, a listing gap, etc.) - reindexing onto
            # this ticker's own dates and forward-filling means every
            # pattern can look up a market-regime value even on a date
            # SPY itself didn't have a fresh bar for.
            ticker_market_regime = market_regime_series.reindex(price_data.index).ffill()
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

            # direction="up", require_full_candle=True - must match
            # label_pattern_outcome()'s own search exactly (see
            # src/labeling.py), or this row's features get computed as of
            # a different day than the one the trade actually enters on.
            # An earlier version of this script searched either direction
            # with a close-only check, which could - and on 22% of rows,
            # did - find an earlier, different breakout than the real
            # entry trigger, silently reintroducing the exact "features
            # read too early" problem this dataset exists to fix (see
            # PLAN.md's Stage 6 notes for a real case on ABT: a 16-day
            # gap between the two).
            breakout = find_breakout_date(
                price_data, pattern, max_search_days=max_search_days, direction="up", require_full_candle=True
            )
            if breakout is None:
                dropped_no_breakout += 1
                continue

            end_date = pattern_evaluation_date(pattern)
            pattern_at_breakout = pattern_as_of_breakout(pattern, breakout["breakout_date"])

            # Stage 3 already computes how *well-formed* a pattern is
            # (trendline fit, contraction, pole strength) but that
            # quality information never reached the model before now - a
            # barely-qualifying pattern and a tight, clean one looked
            # identical. Triangles and bull flags are described by
            # different geometry, so a triangle row gets 0 for the
            # bull-flag-only fields and vice versa - 0 sits clearly
            # outside either field's real range (triangle r² requires
            # >=0.6 to qualify at all; a bull flag's pole_return_pct
            # requires >=15 by construction), so it reads as "not
            # applicable to this pattern type", not as a real low value.
            is_triangle = isinstance(pattern, TrianglePattern)
            geometric_quality = (
                {
                    "pattern_high_r_squared": pattern.high_r_squared,
                    "pattern_low_r_squared": pattern.low_r_squared,
                    "pattern_contraction_pct": pattern.contraction_pct,
                    "pattern_pole_return_pct": 0.0,
                    "pattern_flag_volume_ratio": 0.0,
                    "pattern_retracement_pct": 0.0,
                }
                if is_triangle
                else {
                    "pattern_high_r_squared": 0.0,
                    "pattern_low_r_squared": 0.0,
                    "pattern_contraction_pct": 0.0,
                    "pattern_pole_return_pct": pattern.pole_return_pct,
                    "pattern_flag_volume_ratio": pattern.flag_volume_ratio,
                    "pattern_retracement_pct": pattern.retracement_pct,
                }
            )

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
                **geometric_quality,
                "market_pct_from_50d_average": ticker_market_regime["market_pct_from_50d_average"].loc[breakout["breakout_date"]],
                "market_20d_volatility_pct": ticker_market_regime["market_20d_volatility_pct"].loc[breakout["breakout_date"]],
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
