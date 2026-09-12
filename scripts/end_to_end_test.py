"""
End-to-end smoke test: runs the real pipeline (fetch -> pivots -> triangle/
bull-flag detection -> every Stage 4 indicator combination -> chart
generation) for a ticker, and reports whether anything raised an
exception or produced an obviously broken result (a NaN feature value, or
zero patterns detected).

This is a plain script, not a pytest suite - it hits the live network via
yfinance for real price data, so it's meant to be run by hand as a quick
"did I break anything" check after making changes, not as part of an
automated/offline test run. See COMMANDS.md for how to run it.
"""

import functools
import math
import sys
import tempfile

import main
from scripts.build_breakout_dataset import build_breakout_dataset
from scripts.build_dataset import build_labeled_dataset
from scripts.fetch_real_data import fetch_daily_price_history
from scripts.measure_indicator_lag import measure_indicator_lag
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


def _has_nan_value(features: dict) -> bool:
    """
    True if any numeric value in a feature dict is NaN - a sign an
    indicator's math broke on this data (e.g. a divide-by-zero).
    """
    return any(isinstance(value, float) and math.isnan(value) for value in features.values())


def run_smoke_test(ticker: str = "AAPL", benchmark_ticker: str = "SPY") -> bool:
    """
    Run the full pipeline for `ticker` across every stage and every Stage
    4 indicator combination, printing PASS/FAIL for each check.

    Returns True if every check passed.
    """
    all_passed = True

    def check(label: str, passed: bool) -> None:
        nonlocal all_passed
        print(f"[{'PASS' if passed else 'FAIL'}] {label}")
        if not passed:
            all_passed = False

    print(f"Running end-to-end smoke test on {ticker}...\n")

    print("--- Stages 1-3: data, pivots, pattern detection ---")
    price_data = fetch_daily_price_history(ticker)
    check("fetch_daily_price_history returns data", len(price_data) > 0)

    pivots = find_pivots(price_data)
    check("find_pivots adds swing_high/swing_low columns", {"swing_high", "swing_low"} <= set(pivots.columns))

    triangles = deduplicate_triangles(detect_triangles(pivots))
    bull_flags = deduplicate_bull_flags(detect_bull_flags(price_data))
    bull_flags = remove_bull_flags_inside_wedges(triangles, bull_flags)
    patterns = triangles + bull_flags
    check(
        f"pattern detection finds at least one pattern ({len(triangles)} triangles, {len(bull_flags)} bull flags)",
        len(patterns) > 0,
    )

    wedges = [t for t in triangles if t.triangle_type in ("rising_wedge", "falling_wedge")]
    no_bull_flag_inside_a_wedge = not any(
        wedge.start_date <= bull_flag.pole_start_date and bull_flag.flag_end_date <= wedge.end_date
        for bull_flag in bull_flags
        for wedge in wedges
    )
    check(
        f"remove_bull_flags_inside_wedges leaves no bull flag fully inside a wedge ({len(wedges)} wedges detected)",
        no_bull_flag_inside_a_wedge,
    )

    print("\n--- Stage 3+: real breakout-date detection (find_breakout_date) ---")
    breakout_detection_ok = True
    for pattern in patterns:
        try:
            breakout = find_breakout_date(price_data, pattern)
            if breakout is not None:
                if not ({"breakout_date", "direction"} <= set(breakout.keys()) and breakout["direction"] in ("up", "down")):
                    breakout_detection_ok = False
                # pattern_as_of_breakout() should hand back a copy whose
                # evaluation date is the breakout date, with the original
                # pattern left untouched - a stale value here would mean
                # measure_indicator_lag.py is silently comparing a pattern
                # against itself instead of against its real breakout day.
                pattern_at_breakout = pattern_as_of_breakout(pattern, breakout["breakout_date"])
                if pattern_evaluation_date(pattern_at_breakout) != breakout["breakout_date"]:
                    breakout_detection_ok = False
        except Exception as error:
            print(f"    exception: {error}")
            breakout_detection_ok = False
    check(
        f"find_breakout_date/pattern_as_of_breakout run cleanly on every pattern ({len(patterns)} patterns)",
        breakout_detection_ok,
    )

    print("\n--- Stage 4: indicator feature computation ---")
    benchmark_data = fetch_daily_price_history(benchmark_ticker)
    for indicator_number, combination in INDICATOR_COMBINATIONS.items():
        compute_features = combination["compute_features"]
        if combination.get("needs_benchmark"):
            compute_features = functools.partial(compute_features, benchmark_data=benchmark_data)

        indicator_ok = True
        for pattern in patterns:
            try:
                features = compute_features(price_data, pattern)
                if _has_nan_value(features):
                    indicator_ok = False
            except Exception as error:
                print(f"    exception on indicator #{indicator_number}: {error}")
                indicator_ok = False
        check(f"indicator #{indicator_number} ({combination['name']}) - clean features for every pattern", indicator_ok)

    print("\n--- Stage 5: labeling ---")
    try:
        labels = label_patterns(price_data, patterns)
        labels_ok = all(
            label["exit_reason"] in ("target", "stop", "time") and not math.isnan(label["return_pct"]) for label in labels
        )
        successful_count = sum(label["is_successful"] for label in labels)
        check(f"label_patterns produces clean labels ({len(labels)} of {len(patterns)} patterns, {successful_count} successful)", labels_ok)
    except Exception as error:
        print(f"    exception: {error}")
        check("label_patterns produces clean labels", False)

    print("\n--- Stage 5b: chart generation ---")
    for indicator_number in [None] + sorted(INDICATOR_COMBINATIONS.keys()):
        temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        temp_file.close()
        label = f"chart renders with indicator {indicator_number}" if indicator_number else "chart renders with no indicator"
        try:
            main.run(ticker, indicator_number=indicator_number, save_path=temp_file.name)
            check(label, True)
        except Exception as error:
            print(f"    exception: {error}")
            check(label, False)

    temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
    temp_file.close()
    try:
        main.run(ticker, label_outcomes=True, save_path=temp_file.name)
        check("chart renders with labeled outcomes", True)
    except Exception as error:
        print(f"    exception: {error}")
        check("chart renders with labeled outcomes", False)

    print("\n--- Stage 6: multi-ticker dataset + feature matrix ---")
    try:
        dataset = build_labeled_dataset([ticker], period="2y")
        expected_indicator_columns = {f"indicator{n}_" for n in INDICATOR_COMBINATIONS}
        has_indicator_columns = all(
            any(column.startswith(prefix) for column in dataset.columns) for prefix in expected_indicator_columns
        )
        check(
            f"build_labeled_dataset produces a feature matrix ({len(dataset)} rows, "
            f"{len(dataset.columns)} columns, all 5 indicator prefixes present)",
            len(dataset) > 0 and has_indicator_columns,
        )
    except Exception as error:
        print(f"    exception: {error}")
        check("build_labeled_dataset produces a feature matrix", False)

    print("\n--- Stage 6: indicator-lag measurement ---")
    try:
        measurements = measure_indicator_lag([ticker], period="2y")
        expected_columns = {"confirmation_count_at_end_date", "confirmation_count_at_breakout", "lag_days"}
        check(
            f"measure_indicator_lag compares end_date vs. real-breakout readings ({len(measurements)} patterns "
            f"had a breakout found)",
            expected_columns <= set(measurements.columns),
        )
    except Exception as error:
        print(f"    exception: {error}")
        check("measure_indicator_lag compares end_date vs. real-breakout readings", False)

    print("\n--- Stage 6: breakout-day feature matrix ---")
    try:
        breakout_dataset = build_breakout_dataset([ticker], period="2y")
        expected_indicator_columns = {f"indicator{n}_" for n in INDICATOR_COMBINATIONS}
        has_indicator_columns = all(
            any(column.startswith(prefix) for column in breakout_dataset.columns) for prefix in expected_indicator_columns
        )
        expected_other_columns = {"breakout_date", "breakout_direction", "lag_days"}
        check(
            f"build_breakout_dataset produces a feature matrix read at the real breakout day "
            f"({len(breakout_dataset)} rows, {len(breakout_dataset.columns)} columns)",
            len(breakout_dataset) > 0
            and has_indicator_columns
            and expected_other_columns <= set(breakout_dataset.columns),
        )
    except Exception as error:
        print(f"    exception: {error}")
        check("build_breakout_dataset produces a feature matrix", False)

    print("\n--- Stage 6: model training + walk-forward validation ---")
    try:
        from src.model import load_training_data, train_and_evaluate

        # A single ticker rarely has enough patterns for even one
        # walk-forward fold - use a batch of well-known, liquid tickers
        # instead, purely to exercise the real training code path. Needs
        # to be a big enough batch that every fold sees at least one
        # winning and one losing pattern - a training fold with only one
        # outcome present can't fit a logistic regression at all, which a
        # much smaller sample (e.g. 10 tickers) can hit by bad luck.
        sample_tickers = [
            "AAPL", "MSFT", "AMZN", "GOOGL", "TSLA", "JPM", "XOM", "UNH", "V", "PG",
            "JNJ", "HD", "BAC", "KO", "DIS", "NFLX", "INTC", "CSCO", "PEP", "MRK",
            "ABBV", "CVX", "WMT", "ADBE", "CRM", "NKE", "MCD", "COST", "T", "VZ",
        ]
        model_dataset = build_breakout_dataset(sample_tickers, period="2y")

        temp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        temp_csv.close()
        model_dataset.to_csv(temp_csv.name, index=False)

        features, target, entry_dates = load_training_data(temp_csv.name)
        fold_results = train_and_evaluate(features, target, n_splits=2)
        check(
            f"model trains and validates across walk-forward folds ({len(features)} patterns, "
            f"{len(fold_results)} folds)",
            len(fold_results) == 2 and {"roc_auc", "baseline_win_rate"} <= set(fold_results.columns),
        )
    except Exception as error:
        print(f"    exception: {error}")
        check("model trains and validates across walk-forward folds", False)

    print("\n" + ("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"))
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
