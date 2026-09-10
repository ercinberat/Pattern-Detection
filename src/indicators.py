"""
Stage 4 confirmation indicators: features computed around a detected
pattern to help judge whether a breakout is likely to follow through
rather than fake out.

PLAN.md lists five indicator combinations to build. This file implements
them one function at a time, starting with combination #1 - Bollinger
Band squeeze + volume surge, flagged in the plan as the best fit for
triangles.
"""

import pandas as pd

from src.patterns import BullFlagPattern, TrianglePattern


def _pattern_evaluation_date(pattern):
    """
    The date an indicator should be evaluated at for a given pattern: a
    triangle's end_date, or a bull flag's flag_end_date. Both mean the
    same thing - the last bar of the pattern candidate, where a breakout
    would be expected to happen next.
    """
    if isinstance(pattern, TrianglePattern):
        return pattern.end_date
    if isinstance(pattern, BullFlagPattern):
        return pattern.flag_end_date
    raise TypeError(f"Unrecognized pattern type: {type(pattern)}")


def compute_bollinger_bands(price_data: pd.DataFrame, band_window: int = 20, band_num_std: float = 2.0) -> pd.DataFrame:
    """
    Compute Bollinger Bands for the full price series: a moving average of
    the Close price, plus upper/lower bands set band_num_std standard
    deviations above/below it.

    band_window: how many bars the moving average and standard deviation
        are computed over. 20 is the standard, widely-used setting (about
        one trading month).
    band_num_std: how many standard deviations wide the bands are. 2 is
        the standard setting, capturing roughly 95% of normal price moves.

    Returns a DataFrame (same index as price_data) with columns:
        middle_band, upper_band, lower_band, and band_width_pct (the gap
        between the bands, as a percentage of the middle band - so
        "squeeze" tightness can be compared across different price
        levels).
    """
    middle_band = price_data["Close"].rolling(window=band_window).mean()
    band_std = price_data["Close"].rolling(window=band_window).std()
    upper_band = middle_band + band_num_std * band_std
    lower_band = middle_band - band_num_std * band_std
    band_width_pct = (upper_band - lower_band) / middle_band * 100

    return pd.DataFrame(
        {
            "middle_band": middle_band,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "band_width_pct": band_width_pct,
        }
    )


def bollinger_squeeze_and_volume_surge(
    price_data: pd.DataFrame,
    pattern,
    band_window: int = 20,
    band_num_std: float = 2.0,
    width_percentile_lookback: int = 120,
    squeeze_percentile_threshold: float = 20.0,
    volume_window: int = 20,
    volume_surge_ratio: float = 1.5,
) -> dict:
    """
    Indicator combination #1 from PLAN.md: a Bollinger Band "squeeze"
    (bands unusually narrow relative to their own recent history) plus a
    volume surge, evaluated at a detected pattern's end date.

    A squeeze means price has coiled into a tight range - consistent with
    a triangle contracting toward its apex - and a volume surge alongside
    it is a classic sign that a real breakout is brewing rather than a
    fakeout.

    price_data: DataFrame with Close and Volume columns, e.g. the output
        of fetch_daily_price_history().
    pattern: a TrianglePattern or BullFlagPattern (from src/patterns.py).
        The indicator is evaluated at the pattern's end date.
    band_window, band_num_std: passed straight through to
        compute_bollinger_bands() - see that function for what they mean.
    width_percentile_lookback: how many bars of band-width history to rank
        the current width against, to judge whether it's unusually narrow
        right now rather than just normally narrow.
    squeeze_percentile_threshold: the current band width has to rank below
        this percentile (0-100) of its own recent history to count as a
        "squeeze" - e.g. 20 means today's width is narrower than 80% of
        the last width_percentile_lookback bars.
    volume_window: how many bars the average volume is computed over.
    volume_surge_ratio: current volume has to be at least this many times
        the recent average volume to count as a "surge".

    Returns a dict of feature values:
        band_width_pct: the current band width, as a percentage of the
            moving average price (so it's comparable across price levels).
        band_width_percentile: where that width ranks (0-100) against its
            own recent history - lower means more squeezed.
        is_squeezed: True if band_width_percentile is below
            squeeze_percentile_threshold.
        volume_ratio: current volume divided by the recent average volume.
        is_volume_surge: True if volume_ratio is at least
            volume_surge_ratio.
    """
    evaluation_date = _pattern_evaluation_date(pattern)
    bands = compute_bollinger_bands(price_data, band_window=band_window, band_num_std=band_num_std)

    # Rank today's width against its own trailing history: what fraction
    # of the last width_percentile_lookback bars had a width at or below
    # today's? A low percentile means today's bands are unusually tight.
    width_history = bands["band_width_pct"].loc[:evaluation_date].tail(width_percentile_lookback)
    current_width = bands["band_width_pct"].loc[evaluation_date]
    band_width_percentile = (width_history <= current_width).mean() * 100

    average_volume = price_data["Volume"].rolling(window=volume_window).mean()
    current_volume = price_data["Volume"].loc[evaluation_date]
    volume_ratio = current_volume / average_volume.loc[evaluation_date]

    return {
        "band_width_pct": current_width,
        "band_width_percentile": band_width_percentile,
        "is_squeezed": band_width_percentile < squeeze_percentile_threshold,
        "volume_ratio": volume_ratio,
        "is_volume_surge": volume_ratio >= volume_surge_ratio,
    }


# Maps each indicator combination to its feature function, numbered to
# match PLAN.md's Stage 4 table - lets callers (e.g. main.py's
# --indicator argument) pick a combination by number instead of importing
# a specific function by name. Every combination takes (price_data,
# pattern) and returns a feature dict, so they can all be called the same
# way regardless of which one is selected. As combinations #2-5 are
# built, add them here.
INDICATOR_COMBINATIONS = {
    1: {
        "name": "Bollinger Band squeeze + volume surge",
        "compute_features": bollinger_squeeze_and_volume_surge,
    },
}


if __name__ == "__main__":
    # Quick manual check: fetch AAPL's daily history, detect its patterns,
    # and print this indicator's features for each one, so the numbers
    # can be sanity-checked against the chart by eye.
    from scripts.fetch_real_data import fetch_daily_price_history
    from src.patterns import deduplicate_triangles, detect_bull_flags, detect_triangles, find_pivots

    aapl_daily_data = fetch_daily_price_history("AAPL")
    aapl_pivots = find_pivots(aapl_daily_data, order=5)
    aapl_triangles = deduplicate_triangles(detect_triangles(aapl_pivots))
    aapl_bull_flags = detect_bull_flags(aapl_daily_data)

    for triangle in aapl_triangles:
        features = bollinger_squeeze_and_volume_surge(aapl_daily_data, triangle)
        print(f"{triangle.triangle_type} triangle ending {triangle.end_date.date()}: {features}")

    for bull_flag in aapl_bull_flags:
        features = bollinger_squeeze_and_volume_surge(aapl_daily_data, bull_flag)
        print(f"bull flag ending {bull_flag.flag_end_date.date()}: {features}")
