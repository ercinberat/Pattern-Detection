"""
Pivot detection and geometric pattern detection for daily price data.

Stage 2 (find_pivots) finds swing highs and swing lows. Stage 3
(detect_triangles, detect_bull_flags) uses those swing pivots as the
building blocks for fitting triangle trendlines and finding bull flag
pole+consolidation setups, as described in PLAN.md.
"""

from dataclasses import dataclass, replace

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

    last_high_pivot_date/last_low_pivot_date are the most recent swing
    high/swing low pivot each line was actually fit from. A trendline's
    equation depends on every pivot used to fit it, including ones late
    in the window - so testing whether price has crossed that line on a
    date *before* one of its own defining pivots has even happened yet is
    lookahead bias: the exact line being tested wasn't knowable at that
    point in time. find_breakout_date() uses these two dates to avoid
    exactly that (see PLAN.md's Stage 3 notes for a real case on AAPL,
    where a "breakout" was flagged on a window's very first day using a
    line whose earliest defining pivot didn't happen until 6 days later).
    """

    start_date: pd.Timestamp
    end_date: pd.Timestamp
    triangle_type: str  # "symmetrical", "ascending", "descending", "rising_wedge", or "falling_wedge"
    high_slope: float
    high_intercept: float
    high_r_squared: float
    low_slope: float
    low_intercept: float
    low_r_squared: float
    contraction_pct: float  # how much the high/low trendline gap shrank over the window
    last_high_pivot_date: pd.Timestamp
    last_low_pivot_date: pd.Timestamp


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
    retracement_pct: float  # how much of the pole's gain the flag gave back before this pattern qualified


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


def find_breakout_date(
    price_data: pd.DataFrame,
    pattern,
    max_search_days: int = 20,
    direction: str = None,
    require_full_candle: bool = False,
):
    """
    Find the first bar where price actually breaks the pattern's
    geometry - the real breakout/breakdown candle, as opposed to
    end_date/flag_end_date, which is just the last bar of whichever
    overlapping detection window happened to be kept and can land several
    days away from the real move (see PLAN.md's Stage 3 notes for a real
    example on EXPE, where the window's end_date landed 3 trading days
    after a +17.6% earnings gap).

    A triangle's own fitted trendlines (high_slope/high_intercept and
    low_slope/low_intercept) already describe the exact line a breakout
    has to cross, so "breaking" means Close moving outside those lines.
    The search covers the triangle's own window (start_date onward) as
    well as max_search_days past end_date, not just the days after
    end_date - a real breakout can happen *inside* a window whose
    end_date got pushed past it (exactly what happened on EXPE: the
    window that won deduplicate_triangles()'s tie-break still contained
    the actual gap day, because the fresh pivots created by that gap
    hadn't been confirmed yet - see find_pivots()). Searching inside the
    window also means a candidate date can come before some of the
    pivots a line was fit from - checked against
    last_high_pivot_date/last_low_pivot_date (see TrianglePattern) so a
    cross is never reported before its own line was actually knowable
    (a real case on AAPL flagged a "breakout" on a window's first day,
    using a line whose defining pivots didn't happen until 6 days later -
    see PLAN.md's Stage 3 notes). A bull flag's box is stored directly
    (flag_high/flag_low) and, by construction, can't be broken during its
    own flag_start_date-to-flag_end_date window (Close can never exceed
    flag_high, the max High seen during exactly that window), so its
    search only needs to start after flag_end_date - no separate lookahead
    guard is needed there.

    price_data: DataFrame with a Close column, e.g. fetch_daily_price_history()'s
        output. Must be the same price series the pattern was detected on,
        so bar positions/dates line up with the trendlines' own fit.
    pattern: a TrianglePattern or BullFlagPattern (see above for how each
        one's "break" is defined).
    max_search_days: how many bars past the pattern's own end date to keep
        looking before giving up. Matches labeling.py's default
        max_holding_days, since a breakout that hasn't happened within a
        trade's own holding window isn't useful to compare against.
    direction: None (default) returns the first break in EITHER direction,
        whichever comes first - the right choice for asking "when did this
        pattern's own geometry actually get resolved." "up" or "down"
        instead searches ONLY for a break in that one direction, skipping
        past any opposite-direction break rather than stopping there - the
        right choice for asking "if I only care about a long entry, is
        there a real upward break somewhere in the search range, even if
        an earlier downward wiggle would otherwise have ended the search."
        A triangle's low/high trendline can be a rough, overly steep
        extrapolation (see PLAN.md's Stage 3 notes on TSLA's rising
        wedge), so an early opposite-direction cross doesn't necessarily
        mean the real move never happened - just that it happened later.
    require_full_candle: False (default) checks only the Close price
        against the line, matching every existing use of this function
        (the timing-measurement analyses in PLAN.md's Stage 3/6 notes all
        use this definition, so their published numbers stay reproducible
        with the default). True instead requires the WHOLE candle to have
        cleared the line - Low above the upper line for an "up" break,
        High below the lower line for a "down" break - so a candle that
        dipped below the line intraday but recovered to close above it
        doesn't count. Used by labeling.py's entry_trigger="breakout" for
        a stricter, more conservative entry signal (by explicit request).

    Returns a dict {"breakout_date": pd.Timestamp, "direction": "up" or
    "down"} for the first bar that breaks the pattern's geometry in the
    requested direction (or either, if direction is None), or None if
    that never happens within the search range.
    """
    evaluation_date = pattern_evaluation_date(pattern)
    evaluation_position = price_data.index.get_loc(evaluation_date)
    search_end_position = evaluation_position + max_search_days

    if isinstance(pattern, TrianglePattern):
        # Start from the window's own first bar, not just after end_date -
        # see the docstring above for why a real break can fall inside a
        # window whose end_date has drifted past it.
        start_position = price_data.index.get_loc(pattern.start_date)
        search_dates = price_data.index[start_position : search_end_position + 1]
        for date in search_dates:
            bar_position = price_data.index.get_loc(date)
            upper_line = pattern.high_slope * bar_position + pattern.high_intercept
            lower_line = pattern.low_slope * bar_position + pattern.low_intercept
            # require_full_candle checks the candle's Low/High (the whole
            # candle has to clear the line) instead of just Close (a
            # close-only check still counts a candle that dipped below
            # the line intraday and recovered by the end of the bar).
            up_check_price = price_data["Low"].loc[date] if require_full_candle else price_data["Close"].loc[date]
            down_check_price = price_data["High"].loc[date] if require_full_candle else price_data["Close"].loc[date]
            # A line's equation isn't knowable until every pivot it was
            # fit from has actually happened - checking a cross before
            # that is lookahead bias (see TrianglePattern's docstring for
            # the real AAPL case this guards against). Gated per line,
            # independently: the high line's own last pivot gates an "up"
            # cross, the low line's gates a "down" cross - the other
            # line's pivots aren't relevant to whether this one's
            # equation was already knowable.
            if up_check_price > upper_line and direction != "down" and date >= pattern.last_high_pivot_date:
                return {"breakout_date": date, "direction": "up"}
            if down_check_price < lower_line and direction != "up" and date >= pattern.last_low_pivot_date:
                return {"breakout_date": date, "direction": "down"}
        return None

    if isinstance(pattern, BullFlagPattern):
        # Bull flags only ever check the upside (see the docstring above),
        # so a caller asking for direction="down" can never get a match -
        # that's a deliberate consequence of what a bull flag is, not a
        # bug to special-case here.
        if direction == "down":
            return None
        search_dates = price_data.index[evaluation_position + 1 : search_end_position + 1]
        for date in search_dates:
            up_check_price = price_data["Low"].loc[date] if require_full_candle else price_data["Close"].loc[date]
            if up_check_price > pattern.flag_high:
                return {"breakout_date": date, "direction": "up"}
        return None

    raise TypeError(f"Unrecognized pattern type: {type(pattern)}")


def pattern_as_of_breakout(pattern, breakout_date: pd.Timestamp):
    """
    Return a copy of `pattern` with its evaluation date (end_date for a
    triangle, flag_end_date for a bull flag) moved to breakout_date.

    This is the piece that lets find_breakout_date() plug straight into
    the existing Stage 4 indicator functions and Stage 5 labeling without
    changing either: they all read a pattern's evaluation date via
    pattern_evaluation_date(pattern), so a copy with that one field
    swapped is enough to make them look at "the real breakout day" instead
    of "the detection window's last bar", with every other field
    (trendlines, flag box, etc.) left exactly as detected.
    """
    if isinstance(pattern, TrianglePattern):
        return replace(pattern, end_date=breakout_date)
    if isinstance(pattern, BullFlagPattern):
        return replace(pattern, flag_end_date=breakout_date)
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


# History: min_pivots_per_side was raised from 2 to 3 (requiring 3
# pivots on BOTH sides) after a real case exposed a problem with 2: two
# points always fit a straight line perfectly, r²=1.0, regardless of
# whether the line means anything. EXPE's 2026-04-02 triangle had a "low"
# trendline fit from exactly 2 swing lows, scoring a trivially perfect
# r² while sitting 20+ points away from every actual price low in
# between them (see PLAN.md's Stage 3 notes) - a difference
# deduplicate_triangles()'s r²-based tie-break can't detect, since it
# can't tell "genuinely well-supported line" apart from "only had 2
# points to fit." Requiring 3-on-both turned out to be much stricter
# than intended (EXPE dropped from 3 triangles detected to 0, AAPL from
# 4 to 1), so it was relaxed to the current rule below: both sides still
# need min_pivots_per_side (back to 2), but at least one side must reach
# min_pivots_one_side (3) - min_pivots_per_side=2 is what's superseded
# here, not min_pivots_one_side. Note this narrower rule would NOT have
# rejected the original EXPE case (its high side already had 3 pivots;
# only the low side had 2) - it guards against BOTH sides being a bare
# 2-point line, not against one side being one. Kept here, not just in
# git history, so any of these choices are easy to revisit.
_SUPERSEDED_MIN_PIVOTS_PER_SIDE = 2


def detect_triangles(
    pivots: pd.DataFrame,
    window: int = 40,
    min_pivots_per_side: int = 2,
    min_pivots_one_side: int = 3,
    min_r_squared: float = 0.6,
    min_contraction_pct: float = 25.0,
    flat_slope_threshold_pct: float = 2.0,
) -> list[TrianglePattern]:
    """
    Slide a window across the price data looking for triangle candidates:
    two trendlines (fitted to the swing highs and swing lows) with the gap
    between them shrinking (contracting) over the window - classified as
    "symmetrical" (high declining, low rising), "ascending" (high flat,
    low rising), "descending" (high declining, low flat), "rising_wedge"
    (both rising, low rising faster), or "falling_wedge" (both declining,
    high declining faster). A window where the two lines are flat or
    diverging isn't a triangle candidate at all.

    pivots: output of find_pivots() - price_data plus swing_high/swing_low
        columns.
    window: how many bars to look back over when fitting each candidate's
        trendlines. Needs to be big enough to contain several swings - a
        window of 40 daily bars is roughly two months, which fits the
        swing-trading holding periods this project targets.
    min_pivots_per_side: a window needs at least this many swing highs AND
        this many swing lows to attempt a trendline fit at all - kept at
        the original bare minimum (2) so a real triangle whose flatter
        side only ever gets tapped twice isn't thrown out entirely.
        min_pivots_one_side below is the stricter check that actually
        guards against a degenerate fit.
    min_pivots_one_side: at least one of the two sides (high or low)
        needs this many pivots - so a candidate can't have BOTH sides
        sitting at the bare min_pivots_per_side minimum, which is what
        let a trivially "perfect" 2-point line through in a real case
        (see _SUPERSEDED_MIN_PIVOTS_PER_SIDE above and PLAN.md's Stage 3
        notes). This only guarantees ONE side has real support behind
        it - the other side can still be a 2-point line.
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
        if len(swing_highs) < min_pivots_one_side and len(swing_lows) < min_pivots_one_side:
            continue  # neither side has more than a bare min_pivots_per_side-point line behind it

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
        high_is_rising = high_total_move_pct > flat_slope_threshold_pct
        low_is_flat = abs(low_total_move_pct) < flat_slope_threshold_pct
        low_is_rising = low_total_move_pct > flat_slope_threshold_pct
        low_is_declining = low_total_move_pct < -flat_slope_threshold_pct

        if high_is_declining and low_is_rising:
            triangle_type = "symmetrical"
        elif high_is_flat and low_is_rising:
            triangle_type = "ascending"
        elif high_is_declining and low_is_flat:
            triangle_type = "descending"
        elif high_is_rising and low_is_rising:
            # Both sides rising, but the low side has to be rising faster
            # than the high side for the gap to have contracted at all
            # (already confirmed above) - a "rising wedge", found on TSLA
            # (2025-07-16 -> 2025-09-10, see PLAN.md's Stage 3 notes): a
            # real, valid triangle shape the original three-case
            # classification silently discarded, since it never
            # considered both sides sloping the same direction.
            triangle_type = "rising_wedge"
        elif high_is_declining and low_is_declining:
            # Mirror image of rising_wedge - both sides falling, high side
            # falling faster (again guaranteed by the contraction check
            # above). A "falling wedge".
            triangle_type = "falling_wedge"
        else:
            # Both flat, or diverging - not a converging triangle shape.
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
                # swing_highs/swing_lows are already in chronological order
                # (subsets of window_slice, which is), so the last entry is
                # the most recent pivot each line was actually fit from.
                last_high_pivot_date=swing_highs.index[-1],
                last_low_pivot_date=swing_lows.index[-1],
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
                retracement_pct=retracement_pct,
            )
        )

    return bull_flags


def deduplicate_bull_flags(bull_flags: list[BullFlagPattern]) -> list[BullFlagPattern]:
    """
    Collapse heavily overlapping bull flag candidates down to the single
    best one per overlapping cluster - purely to make charts readable, the
    same problem deduplicate_triangles() solves for triangles.
    detect_bull_flags() checks every bar as a possible pole start, so one
    real pole-and-flag move in the price data can get detected again and
    again from poles starting a few days apart that all land on
    essentially the same flag. detect_bull_flags() itself still returns
    every one of those raw candidates unchanged; this is a separate
    filtering step to run before charting.

    Bull flags are grouped into a cluster whenever their pole-to-flag date
    ranges overlap, and the one with the strongest pole (highest
    pole_return_pct) is kept from each cluster.
    """
    if not bull_flags:
        return []

    sorted_bull_flags = sorted(bull_flags, key=lambda bull_flag: bull_flag.pole_start_date)

    clusters = [[sorted_bull_flags[0]]]
    for bull_flag in sorted_bull_flags[1:]:
        current_cluster = clusters[-1]
        cluster_end_date = max(member.flag_end_date for member in current_cluster)
        if bull_flag.pole_start_date <= cluster_end_date:
            # This candidate's pole-to-flag window overlaps the current
            # cluster's date range, so it's almost certainly the same
            # underlying pole-and-flag move, detected again from a
            # slightly shifted pole start.
            current_cluster.append(bull_flag)
        else:
            clusters.append([bull_flag])

    return [max(cluster, key=lambda bull_flag: bull_flag.pole_return_pct) for cluster in clusters]


def remove_bull_flags_inside_wedges(triangles: list[TrianglePattern], bull_flags: list[BullFlagPattern]) -> list[BullFlagPattern]:
    """
    Drop any bull flag whose whole pole-to-flag date range sits inside a
    detected rising_wedge/falling_wedge - by request, on the reasoning
    that a wedge is a slower, multi-week contraction, so a bull flag's
    much shorter pole-and-flag shape (pole_lookback=10 bars, flag_length=7
    bars by default - a few weeks at most) detected entirely within that
    same window isn't really an independent setup. It's just a smaller
    piece of the same underlying move already being described by the
    wedge, so keeping both would double-count one move as two different
    (and possibly contradictory - e.g. a bullish flag inside a bearish
    falling wedge) pattern types.

    Only rising_wedge/falling_wedge triangles count as "a wedge" here -
    the original symmetrical/ascending/descending triangle types aren't
    considered, since this was raised specifically about wedges.

    triangles: a list of TrianglePattern, e.g. deduplicate_triangles()'s
        output - only its rising_wedge/falling_wedge entries are used.
    bull_flags: a list of BullFlagPattern, e.g. deduplicate_bull_flags()'s
        output.

    Returns a new list: bull_flags with any pattern fully inside a wedge's
    date range removed. Doesn't modify triangles or bull_flags in place.
    """
    wedges = [triangle for triangle in triangles if triangle.triangle_type in ("rising_wedge", "falling_wedge")]

    def is_inside_a_wedge(bull_flag: BullFlagPattern) -> bool:
        return any(
            wedge.start_date <= bull_flag.pole_start_date and bull_flag.flag_end_date <= wedge.end_date
            for wedge in wedges
        )

    return [bull_flag for bull_flag in bull_flags if not is_inside_a_wedge(bull_flag)]


if __name__ == "__main__":
    # Quick manual check: fetch AAPL's daily history, find its pivots, and
    # plot them so we can see the swing highs/lows directly on the chart.
    from scripts.fetch_real_data import fetch_daily_price_history
    from src.charting import plot_chart

    aapl_daily_data = fetch_daily_price_history("AAPL")
    aapl_pivots = find_pivots(aapl_daily_data, order=5)
    plot_chart(aapl_daily_data, ticker="AAPL", pivots=aapl_pivots)
