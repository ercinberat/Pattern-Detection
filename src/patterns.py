"""
Pivot detection and geometric pattern detection for daily price data.

Stage 2 (find_pivots) finds swing highs and swing lows. Stage 3
(detect_triangles, detect_bull_flags) uses those swing pivots as the
building blocks for fitting triangle trendlines and finding bull flag
pole+consolidation setups, as described in PLAN.md.
"""

from dataclasses import dataclass

import pandas as pd
from scipy.stats import linregress


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


@dataclass
class TrianglePattern:
    """
    A detected triangle: two converging trendlines fitted to the swing
    highs and swing lows within a window of bars.

    high_slope/high_intercept and low_slope/low_intercept describe the
    fitted trendline equations (price = slope * bar_position + intercept,
    where bar_position is the bar's position - 0, 1, 2, ... - within the
    full price series). Storing the fitted equations directly, rather than
    just start/end points, means the exact fitted line can always be
    redrawn or re-evaluated later, at any point along it.
    """

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    triangle_type: str  # "symmetrical", "ascending", or "descending"
    high_slope: float
    high_intercept: float
    high_r_squared: float
    low_slope: float
    low_intercept: float
    low_r_squared: float
    contraction_pct: float  # how much the high/low trendline gap shrank over the window


@dataclass
class BullFlagPattern:
    """
    A detected bullish flag: a sharp upward "pole" move, followed
    immediately by a tight, low-volume "flag" consolidation.
    """

    pole_start_date: pd.Timestamp
    pole_end_date: pd.Timestamp
    flag_start_date: pd.Timestamp
    flag_end_date: pd.Timestamp
    flag_high: float
    flag_low: float
    pole_return_pct: float
    flag_range_pct: float
    flag_volume_ratio: float


def pattern_evaluation_date(pattern) -> pd.Timestamp:
    """
    The date a detected pattern's outcome should be evaluated from: a
    triangle's end_date, or a bull flag's flag_end_date. Both mean the
    same thing - the last bar of the pattern candidate, where a breakout
    would be expected to happen next. Used by both indicators.py (Stage 4)
    and labeling.py (Stage 5) to line up their calculations with the same
    reference point on a pattern.
    """
    if isinstance(pattern, TrianglePattern):
        return pattern.end_date
    if isinstance(pattern, BullFlagPattern):
        return pattern.flag_end_date
    raise TypeError(f"Unrecognized pattern type: {type(pattern)}")


def _fit_trendline(bar_positions, prices):
    """
    Fit a straight line through a set of (bar_position, price) points
    using least-squares regression.

    Returns the line's slope, intercept, and r-squared - how well the
    points actually fit a straight line, from 0 (no linear relationship)
    to 1 (a perfect fit).
    """
    slope, intercept, correlation, _p_value, _std_err = linregress(bar_positions, prices)
    r_squared = correlation**2
    return slope, intercept, r_squared


def detect_triangles(
    pivots: pd.DataFrame,
    window: int = 40,
    min_pivots_per_side: int = 2,
    min_r_squared: float = 0.6,
    min_contraction_pct: float = 25.0,
    flat_slope_threshold_pct: float = 2.0,
) -> list[TrianglePattern]:
    """
    Slide a window across the price data looking for triangle candidates:
    a trendline fitted to the swing highs that's flat or declining, and a
    trendline fitted to the swing lows that's flat or rising, with the gap
    between the two lines shrinking (contracting) over the window.

    pivots: output of find_pivots() - price_data plus swing_high/swing_low
        columns.
    window: how many bars to look back over when fitting each candidate's
        trendlines. Needs to be big enough to contain several swings - a
        window of 40 daily bars is roughly two months, which fits the
        swing-trading holding periods this project targets.
    min_pivots_per_side: a window needs at least this many swing highs AND
        this many swing lows to attempt a trendline fit - two points
        already make a line, but a couple more make the fit meaningful
        rather than just connecting two dots.
    min_r_squared: how well each trendline has to fit its swing points
        (0-1) to count as a genuine trendline rather than a scattered set
        of points that happen to have some best-fit slope.
    min_contraction_pct: how much the gap between the two trendlines must
        shrink from the start of the window to the end, as a percentage of
        the starting gap, to count as "converging" rather than roughly
        parallel or diverging.
    flat_slope_threshold_pct: a trendline counts as "flat" if the total
        price move it implies over the whole window is smaller than this
        percentage of the window's average price - used to tell an
        ascending/descending triangle's flat side apart from its sloped
        side.

    Returns a list of TrianglePattern, one per window that passed all the
    checks. Overlapping windows can each produce their own candidate -
    this function finds raw geometric candidates, it doesn't merge
    duplicates or confirm a breakout (that's Stage 4's job).
    """
    triangles = []

    # The x-axis for the trendline fits is each bar's position (0, 1,
    # 2, ...) within the full price series, since linregress needs plain
    # numbers, not timestamps, and using an absolute position (rather than
    # a position relative to each window) means the same fitted line
    # equation still applies if it's ever evaluated outside its window.
    bar_position = pd.Series(range(len(pivots)), index=pivots.index)

    for window_end in range(window, len(pivots)):
        window_start = window_end - window
        window_slice = pivots.iloc[window_start:window_end]

        swing_highs = window_slice.dropna(subset=["swing_high"])
        swing_lows = window_slice.dropna(subset=["swing_low"])
        if len(swing_highs) < min_pivots_per_side or len(swing_lows) < min_pivots_per_side:
            continue

        high_slope, high_intercept, high_r_squared = _fit_trendline(
            bar_position.loc[swing_highs.index], swing_highs["swing_high"]
        )
        low_slope, low_intercept, low_r_squared = _fit_trendline(
            bar_position.loc[swing_lows.index], swing_lows["swing_low"]
        )
        if high_r_squared < min_r_squared or low_r_squared < min_r_squared:
            continue

        # Evaluate both trendlines at the start and end of the window, to
        # check that they're actually converging rather than parallel,
        # diverging, or already crossed.
        gap_at_start = (high_slope * window_start + high_intercept) - (low_slope * window_start + low_intercept)
        gap_at_end = (high_slope * (window_end - 1) + high_intercept) - (low_slope * (window_end - 1) + low_intercept)
        if gap_at_start <= 0 or gap_at_end <= 0:
            continue  # the lines have already crossed somewhere in the window
        contraction_pct = (1 - gap_at_end / gap_at_start) * 100
        if contraction_pct < min_contraction_pct:
            continue

        average_price = window_slice["Close"].mean()
        high_total_move_pct = high_slope * window / average_price * 100
        low_total_move_pct = low_slope * window / average_price * 100

        high_is_flat = abs(high_total_move_pct) < flat_slope_threshold_pct
        high_is_declining = high_total_move_pct < -flat_slope_threshold_pct
        low_is_flat = abs(low_total_move_pct) < flat_slope_threshold_pct
        low_is_rising = low_total_move_pct > flat_slope_threshold_pct

        if high_is_declining and low_is_rising:
            triangle_type = "symmetrical"
        elif high_is_flat and low_is_rising:
            triangle_type = "ascending"
        elif high_is_declining and low_is_flat:
            triangle_type = "descending"
        else:
            # Both flat, or sloping the wrong way to be converging (e.g.
            # highs rising) - not a triangle shape.
            continue

        triangles.append(
            TrianglePattern(
                start_date=window_slice.index[0],
                end_date=window_slice.index[-1],
                triangle_type=triangle_type,
                high_slope=high_slope,
                high_intercept=high_intercept,
                high_r_squared=high_r_squared,
                low_slope=low_slope,
                low_intercept=low_intercept,
                low_r_squared=low_r_squared,
                contraction_pct=contraction_pct,
            )
        )

    return triangles


def deduplicate_triangles(triangles: list[TrianglePattern]) -> list[TrianglePattern]:
    """
    Collapse heavily overlapping triangle candidates down to the single
    best-fitting one per overlapping cluster - purely to make charts
    readable. Because detect_triangles() slides its window one bar at a
    time, one real triangle shape in the price data gets detected again
    and again from dozens of slightly shifted, overlapping windows.
    detect_triangles() itself still returns every one of those raw
    candidates unchanged; this is a separate filtering step to run before
    charting (or before Stage 4 confirmation, if the same clutter turns
    out to matter there too).

    Triangles are grouped into a cluster whenever their date ranges
    overlap, and the triangle with the best combined trendline fit
    (highest high_r_squared + low_r_squared) is kept from each cluster.
    """
    if not triangles:
        return []

    sorted_triangles = sorted(triangles, key=lambda triangle: triangle.start_date)

    clusters = [[sorted_triangles[0]]]
    for triangle in sorted_triangles[1:]:
        current_cluster = clusters[-1]
        cluster_end_date = max(member.end_date for member in current_cluster)
        if triangle.start_date <= cluster_end_date:
            # This candidate's window overlaps the current cluster's date
            # range, so it's almost certainly the same underlying triangle
            # shape, detected again from a slightly shifted window.
            current_cluster.append(triangle)
        else:
            clusters.append([triangle])

    return [max(cluster, key=lambda t: t.high_r_squared + t.low_r_squared) for cluster in clusters]


def detect_bull_flags(
    price_data: pd.DataFrame,
    pole_lookback: int = 10,
    pole_min_return_pct: float = 15.0,
    flag_length: int = 7,
    flag_max_range_pct: float = 8.0,
    flag_max_volume_ratio: float = 0.7,
    flag_max_retracement_pct: float = 50.0,
) -> list[BullFlagPattern]:
    """
    Slide across the price data looking for bull flag candidates: a sharp
    upward "pole" move over `pole_lookback` bars, immediately followed by
    a tight, low-volume "flag" consolidation lasting `flag_length` bars.

    pole_lookback: how many bars the pole move is measured over. A
        swing-trading pole is a fast move, so this is deliberately short -
        about two trading weeks.
    pole_min_return_pct: the minimum % price gain from the start to the
        end of the pole window to count as a "sharp" move rather than
        ordinary drift.
    flag_length: how many bars the flag consolidation lasts. Fixed rather
        than searched over a range, to keep the detection logic simple -
        real flags will vary somewhat in length around this.
    flag_max_range_pct: the flag's high-to-low range, as a percentage of
        the price at the end of the pole, must stay within this to count
        as a "tight" consolidation rather than continued volatile trading.
    flag_max_volume_ratio: the flag's average volume, as a fraction of the
        pole's average volume, must stay below this - volume should dry up
        during a healthy consolidation.
    flag_max_retracement_pct: the flag can't give back more than this
        percentage of the pole's price gain - a flag that retraces too
        much of the pole isn't holding its gains, which makes it a weaker
        setup.

    Returns a list of BullFlagPattern, one per pole+flag combination that
    passed all the checks.
    """
    bull_flags = []

    for pole_end_index in range(pole_lookback, len(price_data) - flag_length):
        pole_start_index = pole_end_index - pole_lookback
        pole_start_close = price_data["Close"].iloc[pole_start_index]
        pole_end_close = price_data["Close"].iloc[pole_end_index]
        pole_return_pct = (pole_end_close - pole_start_close) / pole_start_close * 100
        if pole_return_pct < pole_min_return_pct:
            continue

        pole_slice = price_data.iloc[pole_start_index : pole_end_index + 1]
        flag_slice = price_data.iloc[pole_end_index + 1 : pole_end_index + 1 + flag_length]

        flag_range_pct = (flag_slice["High"].max() - flag_slice["Low"].min()) / pole_end_close * 100
        if flag_range_pct > flag_max_range_pct:
            continue

        flag_volume_ratio = flag_slice["Volume"].mean() / pole_slice["Volume"].mean()
        if flag_volume_ratio > flag_max_volume_ratio:
            continue

        pole_gain = pole_end_close - pole_start_close
        retracement_pct = (pole_end_close - flag_slice["Low"].min()) / pole_gain * 100
        if retracement_pct > flag_max_retracement_pct:
            continue

        bull_flags.append(
            BullFlagPattern(
                pole_start_date=price_data.index[pole_start_index],
                pole_end_date=price_data.index[pole_end_index],
                flag_start_date=flag_slice.index[0],
                flag_end_date=flag_slice.index[-1],
                flag_high=flag_slice["High"].max(),
                flag_low=flag_slice["Low"].min(),
                pole_return_pct=pole_return_pct,
                flag_range_pct=flag_range_pct,
                flag_volume_ratio=flag_volume_ratio,
            )
        )

    return bull_flags


if __name__ == "__main__":
    # Quick manual check: fetch AAPL's daily history, find its pivots, and
    # plot them so we can see the swing highs/lows directly on the chart.
    from scripts.fetch_real_data import fetch_daily_price_history
    from src.charting import plot_chart

    aapl_daily_data = fetch_daily_price_history("AAPL")
    aapl_pivots = find_pivots(aapl_daily_data, order=5)
    plot_chart(aapl_daily_data, ticker="AAPL", pivots=aapl_pivots)
