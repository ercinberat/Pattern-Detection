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
from scripts.fetch_real_data import fetch_daily_price_history
from src.indicators import INDICATOR_COMBINATIONS
from src.labeling import label_patterns
from src.patterns import deduplicate_triangles, detect_bull_flags, detect_triangles, find_pivots


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
    bull_flags = detect_bull_flags(price_data)
    patterns = triangles + bull_flags
    check(
        f"pattern detection finds at least one pattern ({len(triangles)} triangles, {len(bull_flags)} bull flags)",
        len(patterns) > 0,
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

    print("\n" + ("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED"))
    return all_passed


if __name__ == "__main__":
    success = run_smoke_test()
    sys.exit(0 if success else 1)
