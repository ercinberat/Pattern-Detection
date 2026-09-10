"""
Pivot detection: finding swing highs and swing lows in daily price data.

This is Stage 2 of the pipeline described in PLAN.md. Swing pivots are the
building blocks the later pattern-detection stage (triangles, bull flags)
will fit trendlines to.
"""

import pandas as pd


def find_pivots(price_data: pd.DataFrame, order: int = 5) -> pd.DataFrame:
    """
    Find swing highs and swing lows using the "fractal" method: a bar is a
    swing high if its High price is greater than the High price of every
    bar within `order` days before AND after it. A swing low is the mirror
    image, using Low prices.

    order: how many bars on each side must be lower/higher for a bar to
        count as a pivot. A bigger order finds fewer, more significant
        swings (good for the bigger moves triangles/flags are built from);
        a smaller order finds more, smaller swings and is noisier.

    Returns a copy of price_data with two new columns added:
        swing_high: the High price on bars that are a swing high, NaN on
            every other bar.
        swing_low: the Low price on bars that are a swing low, NaN on
            every other bar.
    """
    pivots = price_data.copy()

    # Start by assuming every bar could be a pivot, then rule bars out as
    # we check them against their neighbors.
    is_swing_high = pd.Series(True, index=pivots.index)
    is_swing_low = pd.Series(True, index=pivots.index)

    # Compare each bar to the bar `offset` days before it and `offset` days
    # after it, for every offset from 1 up to `order`. A bar only survives
    # as a swing high/low if it beats ALL of its neighbors on both sides.
    for offset in range(1, order + 1):
        earlier_high = pivots["High"].shift(offset)
        later_high = pivots["High"].shift(-offset)
        is_swing_high &= (pivots["High"] > earlier_high) & (pivots["High"] > later_high)

        earlier_low = pivots["Low"].shift(offset)
        later_low = pivots["Low"].shift(-offset)
        is_swing_low &= (pivots["Low"] < earlier_low) & (pivots["Low"] < later_low)

    # Note: bars within `order` days of the start/end of the data don't
    # have a full window of neighbors on both sides. shift() gives NaN for
    # those missing neighbors, and comparing anything to NaN evaluates to
    # False in pandas, so those edge bars are correctly never marked as
    # pivots rather than raising an error or giving a false positive.

    pivots["swing_high"] = pivots["High"].where(is_swing_high)
    pivots["swing_low"] = pivots["Low"].where(is_swing_low)

    return pivots


if __name__ == "__main__":
    # Quick manual check: fetch AAPL's daily history, find its pivots, and
    # plot them so we can see the swing highs/lows directly on the chart.
    from scripts.fetch_real_data import fetch_daily_price_history
    from src.charting import plot_chart

    aapl_daily_data = fetch_daily_price_history("AAPL")
    aapl_pivots = find_pivots(aapl_daily_data, order=5)
    plot_chart(aapl_daily_data, ticker="AAPL", pivots=aapl_pivots)
