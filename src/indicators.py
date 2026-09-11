"""
Stage 4 confirmation indicators: features computed around a detected
pattern to help judge whether a breakout is likely to follow through
rather than fake out.

PLAN.md lists five indicator combinations to build. This file implements
them one function at a time:
  #1 - Bollinger Band squeeze + volume surge (best fit for triangles)
  #2 - ADX/DMI trend filter + MACD histogram flip (general gate, both
       patterns)
  #3 - Donchian Channel breakout + OBV confirming new high (best fit for
       flag continuations)
  #4 - RSI momentum shift + ATR expansion off a low (best fit for
       triangles/flags after a basing period)
  #5 - Proximity to 52-week high + relative strength vs. a benchmark
       (best fit for flags after a strong uptrend)

Every combination except #5 only needs the ticker's own price data.
Combination #5 needs a second ticker's data (an index/sector ETF to
compare against), so it doesn't fit the plain (price_data, pattern) shape
the others share - see its docstring below for how callers handle that.
"""

import pandas as pd

from src.patterns import BullFlagPattern, TrianglePattern, pattern_evaluation_date


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
    evaluation_date = pattern_evaluation_date(pattern)
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


def compute_adx_dmi(price_data: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Compute the Average Directional Index (ADX) and its two directional
    components (+DI, -DI) using Wilder's smoothing method - the standard
    way ADX/DMI is calculated.

    ADX measures how strongly a market is trending (in either direction),
    from 0 (no trend) up to 100 (extremely strong trend) - it doesn't say
    which direction. +DI and -DI measure upward vs. downward directional
    movement; comparing them says which direction the trend (if any) is
    actually in.

    window: how many bars the smoothing is done over. 14 is the standard,
        widely-used setting.

    Returns a DataFrame (same index as price_data) with columns: adx,
    plus_di, minus_di.
    """
    high = price_data["High"]
    low = price_data["Low"]
    close = price_data["Close"]

    # A bar's "up move" and "down move" are how much its high/low pushed
    # beyond the previous bar's. Only the larger, positive one of the two
    # counts toward that day's directional movement - a day can't be
    # trending up and down at the same time.
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)

    # Wilder's smoothing is a specific exponential moving average with
    # alpha = 1/window; pandas' ewm(adjust=False) implements exactly that
    # recursive formula, so it's a drop-in way to do Wilder smoothing
    # without hand-rolling the recursion.
    smoothed_true_range = true_range.ewm(alpha=1 / window, adjust=False).mean()
    smoothed_plus_dm = plus_dm.ewm(alpha=1 / window, adjust=False).mean()
    smoothed_minus_dm = minus_dm.ewm(alpha=1 / window, adjust=False).mean()

    plus_di = 100 * smoothed_plus_dm / smoothed_true_range
    minus_di = 100 * smoothed_minus_dm / smoothed_true_range

    directional_index = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = directional_index.ewm(alpha=1 / window, adjust=False).mean()

    return pd.DataFrame({"adx": adx, "plus_di": plus_di, "minus_di": minus_di})


def compute_macd(
    price_data: pd.DataFrame,
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence/Divergence): the gap between
    a fast and a slow exponential moving average of Close price (the
    "MACD line"), a smoothed version of that line (the "signal line"),
    and the difference between them (the "histogram").

    The histogram crossing from negative to positive is a classic bullish
    momentum-shift signal - it means the MACD line has just crossed above
    its own signal line.

    fast_window, slow_window: how many bars the two EMAs are computed
        over. 12/26 are the standard, widely-used settings.
    signal_window: how many bars the signal line's EMA is computed over.
        9 is the standard setting.

    Returns a DataFrame (same index as price_data) with columns:
        macd_line, signal_line, histogram.
    """
    fast_ema = price_data["Close"].ewm(span=fast_window, adjust=False).mean()
    slow_ema = price_data["Close"].ewm(span=slow_window, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_window, adjust=False).mean()
    histogram = macd_line - signal_line

    return pd.DataFrame({"macd_line": macd_line, "signal_line": signal_line, "histogram": histogram})


def adx_trend_filter_and_macd_flip(
    price_data: pd.DataFrame,
    pattern,
    adx_window: int = 14,
    adx_trend_threshold: float = 20.0,
    macd_fast_window: int = 12,
    macd_slow_window: int = 26,
    macd_signal_window: int = 9,
    macd_flip_lookback: int = 5,
) -> dict:
    """
    Indicator combination #2 from PLAN.md: an ADX/DMI trend filter plus a
    MACD histogram flip, evaluated at a detected pattern's end date.
    Flagged in the plan as a general gate for both triangles and bull
    flags, rather than being specific to one pattern shape.

    The ADX/DMI side checks that a real trend actually exists (ADX above
    adx_trend_threshold) and that it's pointed the right way (+DI above
    -DI, i.e. upward) - a breakout without an underlying trend is more
    likely to be noise. The MACD side checks for a recent bullish
    momentum shift (the histogram flipping from negative to positive
    within the last macd_flip_lookback bars) as a timing trigger.

    price_data: DataFrame with High, Low, and Close columns, e.g. the
        output of fetch_daily_price_history().
    pattern: a TrianglePattern or BullFlagPattern (from src/patterns.py).
        The indicator is evaluated at the pattern's end date.
    adx_window: passed straight through to compute_adx_dmi().
    adx_trend_threshold: the ADX value has to be at least this high to
        count as "trending" rather than a directionless chop - 20 is a
        commonly used cutoff for "a trend exists".
    macd_fast_window, macd_slow_window, macd_signal_window: passed
        straight through to compute_macd().
    macd_flip_lookback: how many recent bars to check for a bullish
        histogram flip over. A short window, since a "flip" should be a
        recent event, not something that happened long ago.

    Returns a dict of feature values:
        adx: the current ADX value.
        plus_di, minus_di: the current directional index values.
        is_trending: True if adx is at least adx_trend_threshold.
        is_bullish_direction: True if plus_di is above minus_di.
        macd_histogram: the current MACD histogram value.
        is_macd_bullish_flip: True if the histogram was negative at the
            start of the lookback window and positive now.
    """
    evaluation_date = pattern_evaluation_date(pattern)

    adx_dmi = compute_adx_dmi(price_data, window=adx_window)
    macd = compute_macd(
        price_data,
        fast_window=macd_fast_window,
        slow_window=macd_slow_window,
        signal_window=macd_signal_window,
    )

    current_adx = adx_dmi["adx"].loc[evaluation_date]
    current_plus_di = adx_dmi["plus_di"].loc[evaluation_date]
    current_minus_di = adx_dmi["minus_di"].loc[evaluation_date]

    # Only checking the histogram's value at the start and end of the
    # lookback window (rather than every bar in between) is a
    # simplification - it catches a flip that happened somewhere in the
    # window even if it isn't the very latest bar-to-bar change.
    recent_histogram = macd["histogram"].loc[:evaluation_date].tail(macd_flip_lookback)
    current_histogram = recent_histogram.iloc[-1]
    is_macd_bullish_flip = recent_histogram.iloc[0] < 0 and current_histogram > 0

    return {
        "adx": current_adx,
        "plus_di": current_plus_di,
        "minus_di": current_minus_di,
        "is_trending": current_adx >= adx_trend_threshold,
        "is_bullish_direction": current_plus_di > current_minus_di,
        "macd_histogram": current_histogram,
        "is_macd_bullish_flip": is_macd_bullish_flip,
    }


def compute_donchian_channel(price_data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute the Donchian Channel: the highest High and lowest Low over a
    rolling window of prior bars, not counting the current bar itself.

    Excluding the current bar matters - without it, today's High would
    always be less-than-or-equal to a rolling max that includes today, so
    a "breakout above the upper channel" could never actually happen. By
    shifting the rolling window back one bar, the upper channel instead
    means "the highest high over the window bars *before* today", which
    today's close can genuinely break above.

    window: how many prior bars the channel is measured over. 20 is the
        standard setting (about one trading month).

    Returns a DataFrame (same index as price_data) with columns:
        upper_channel, lower_channel.
    """
    upper_channel = price_data["High"].rolling(window=window).max().shift(1)
    lower_channel = price_data["Low"].rolling(window=window).min().shift(1)
    return pd.DataFrame({"upper_channel": upper_channel, "lower_channel": lower_channel})


def compute_obv(price_data: pd.DataFrame) -> pd.Series:
    """
    Compute On-Balance Volume (OBV): a running total of volume that adds
    the day's volume on an up day, subtracts it on a down day, and holds
    steady on a flat day.

    The idea: rising OBV means volume is flowing in on up days more than
    it's flowing out on down days - a sign that real buying, not just
    noise, is behind a price move. If price makes a new high but OBV
    doesn't, that's a divergence suggesting the move isn't backed by real
    participation.

    Returns a Series (same index as price_data) of the cumulative OBV
    value.
    """
    price_change = price_data["Close"].diff()
    # +1 on an up day, -1 on a down day, 0 on a flat day (or the first
    # bar, which has no prior close to compare against).
    price_direction = price_change.apply(lambda change: 1 if change > 0 else (-1 if change < 0 else 0))
    return (price_direction * price_data["Volume"]).cumsum()


def donchian_breakout_and_obv_confirmation(
    price_data: pd.DataFrame,
    pattern,
    donchian_window: int = 20,
    obv_new_high_lookback: int = 20,
) -> dict:
    """
    Indicator combination #3 from PLAN.md: a Donchian Channel breakout
    plus OBV confirming a new high, evaluated at a detected pattern's end
    date. Flagged in the plan as the best fit for flag continuations - a
    bull flag "works" when price breaks above its recent range on real
    volume, continuing the pole's move rather than stalling out.

    price_data: DataFrame with High, Low, Close, and Volume columns.
    pattern: a TrianglePattern or BullFlagPattern (from src/patterns.py).
        The indicator is evaluated at the pattern's end date.
    donchian_window: passed straight through to compute_donchian_channel().
    obv_new_high_lookback: how many prior bars of OBV history to check
        for a new high over, not counting today - matches
        donchian_window's scale by default, for the same reason (compare
        today against the recent past, not including today).

    Returns a dict of feature values:
        close: the pattern's end-date Close price.
        donchian_upper: the breakout level - the highest High over the
            donchian_window bars before today.
        is_donchian_breakout: True if close is above donchian_upper.
        obv: the current OBV value.
        obv_recent_high: the highest OBV value over the
            obv_new_high_lookback bars before today.
        is_obv_new_high: True if obv is above obv_recent_high - OBV
            itself just broke out to a new high alongside price.
    """
    evaluation_date = pattern_evaluation_date(pattern)

    channel = compute_donchian_channel(price_data, window=donchian_window)
    obv_series = compute_obv(price_data)
    obv_recent_high_series = obv_series.rolling(window=obv_new_high_lookback).max().shift(1)

    current_close = price_data["Close"].loc[evaluation_date]
    donchian_upper = channel["upper_channel"].loc[evaluation_date]
    current_obv = obv_series.loc[evaluation_date]
    obv_recent_high = obv_recent_high_series.loc[evaluation_date]

    return {
        "close": current_close,
        "donchian_upper": donchian_upper,
        "is_donchian_breakout": current_close > donchian_upper,
        "obv": current_obv,
        "obv_recent_high": obv_recent_high,
        "is_obv_new_high": current_obv > obv_recent_high,
    }


def compute_rsi(price_data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute RSI (Relative Strength Index) using Wilder's smoothing - the
    standard way RSI is calculated. RSI compares the size of recent gains
    to recent losses, scaled to 0-100: above 50 means gains have
    outweighed losses recently (bullish momentum), below 50 means the
    opposite.

    window: how many bars the smoothing is done over. 14 is the standard,
        widely-used setting.
    """
    price_change = price_data["Close"].diff()
    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    # Same Wilder-smoothing approach used for ADX/DMI - ewm(adjust=False)
    # with alpha=1/window implements Wilder's recursive formula exactly.
    average_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    average_loss = loss.ewm(alpha=1 / window, adjust=False).mean()

    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def compute_atr(price_data: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Compute ATR (Average True Range) using Wilder's smoothing - a measure
    of how much a stock is moving, up or down, regardless of direction.

    A quiet, "coiled" stock has a low ATR; a stock in the middle of a big
    move has a high one. ATR rising after a stretch of low ATR
    ("volatility contraction") is a classic VCP-style sign that a quiet
    setup is starting to expand into a real move.

    window: how many bars the smoothing is done over. 14 is the standard,
        widely-used setting.
    """
    high = price_data["High"]
    low = price_data["Low"]
    previous_close = price_data["Close"].shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False).mean()


def rsi_momentum_shift_and_atr_expansion(
    price_data: pd.DataFrame,
    pattern,
    rsi_window: int = 14,
    rsi_basing_threshold: float = 50.0,
    rsi_momentum_threshold: float = 57.5,
    rsi_basing_lookback: int = 20,
    atr_window: int = 14,
    atr_low_lookback: int = 20,
    atr_expansion_ratio: float = 1.2,
) -> dict:
    """
    Indicator combination #4 from PLAN.md: an RSI momentum shift (RSI was
    "basing" below rsi_basing_threshold recently, and has since crossed
    above rsi_momentum_threshold) plus an ATR expansion off a recent low
    (a VCP-style volatility contraction followed by expansion), evaluated
    at a detected pattern's end date. Flagged in the plan as fitting
    triangles/flags that follow a basing period particularly well.

    price_data: DataFrame with High, Low, and Close columns.
    pattern: a TrianglePattern or BullFlagPattern (from src/patterns.py).
        The indicator is evaluated at the pattern's end date.
    rsi_window: passed straight through to compute_rsi().
    rsi_basing_threshold: RSI has to have dipped below this value at some
        point in the recent past to count as having been "basing" -
        neutral-to-weak momentum, not already trending. 50 is RSI's own
        neutral midpoint.
    rsi_momentum_threshold: RSI has to be above this value now to count
        as a momentum shift. 57.5 is the midpoint of the 55-60 range
        PLAN.md calls out.
    rsi_basing_lookback: how many bars back to check for that basing dip.
    atr_window: passed straight through to compute_atr().
    atr_low_lookback: how many prior bars to find the recent ATR low
        over, not counting today (mirrors the Donchian Channel's
        exclude-today logic, for the same reason).
    atr_expansion_ratio: current ATR has to be at least this many times
        the recent ATR low to count as "expanding".

    Returns a dict of feature values:
        rsi: the current RSI value.
        was_basing: True if RSI dipped below rsi_basing_threshold at any
            point in the rsi_basing_lookback bars before today.
        is_momentum_shift: True if was_basing and rsi is now above
            rsi_momentum_threshold.
        atr: the current ATR value.
        atr_recent_low: the lowest ATR value over the atr_low_lookback
            bars before today.
        is_atr_expanding: True if atr is at least atr_expansion_ratio
            times atr_recent_low.
    """
    evaluation_date = pattern_evaluation_date(pattern)

    rsi_series = compute_rsi(price_data, window=rsi_window)
    atr_series = compute_atr(price_data, window=atr_window)
    atr_recent_low_series = atr_series.rolling(window=atr_low_lookback).min().shift(1)

    rsi_history = rsi_series.loc[:evaluation_date].tail(rsi_basing_lookback)
    current_rsi = rsi_series.loc[evaluation_date]
    was_basing = bool((rsi_history < rsi_basing_threshold).any())

    current_atr = atr_series.loc[evaluation_date]
    atr_recent_low = atr_recent_low_series.loc[evaluation_date]

    return {
        "rsi": current_rsi,
        "was_basing": was_basing,
        "is_momentum_shift": was_basing and current_rsi > rsi_momentum_threshold,
        "atr": current_atr,
        "atr_recent_low": atr_recent_low,
        "is_atr_expanding": current_atr >= atr_recent_low * atr_expansion_ratio,
    }


def compute_pct_from_52_week_high(price_data: pd.DataFrame, window: int = 252) -> pd.Series:
    """
    Compute how far today's Close sits below its own trailing 252-day
    (~52-week) high, as a percentage. 0% means today's Close IS the
    52-week high; 10% means today's Close is 10% below the highest close
    in roughly the past year.

    window: how many bars count as "52 weeks". 252 is the standard
        approximation for a year of trading days.
    """
    rolling_high = price_data["Close"].rolling(window=window, min_periods=1).max()
    return (rolling_high - price_data["Close"]) / rolling_high * 100


def compute_relative_strength(price_data: pd.DataFrame, benchmark_data: pd.DataFrame, lookback: int = 63) -> pd.Series:
    """
    Compute relative strength: the ticker's return over the past
    `lookback` bars, divided by a benchmark's (e.g. an index or sector
    ETF) return over the same bars. Above 1.0 means the ticker has
    outperformed the benchmark over that stretch - real relative
    strength/leadership, not just moving with the broader market.

    benchmark_data: daily OHLCV DataFrame for the benchmark, e.g. also
        from fetch_daily_price_history(). Its dates are aligned to
        price_data's via reindex, in case the two tickers' trading
        calendars don't match up exactly.
    lookback: how many bars the return is measured over. 63 is roughly
        one trading quarter.
    """
    ticker_return = price_data["Close"] / price_data["Close"].shift(lookback)
    benchmark_close = benchmark_data["Close"].reindex(price_data.index).ffill()
    benchmark_return = benchmark_close / benchmark_close.shift(lookback)
    return ticker_return / benchmark_return


def near_52_week_high_and_relative_strength(
    price_data: pd.DataFrame,
    pattern,
    benchmark_data: pd.DataFrame,
    proximity_window: int = 252,
    max_pct_from_high: float = 15.0,
    relative_strength_lookback: int = 63,
) -> dict:
    """
    Indicator combination #5 from PLAN.md: proximity to the 52-week high
    plus relative strength vs. a benchmark (index/sector ETF), evaluated
    at a detected pattern's end date. Flagged in the plan as fitting
    flags after a strong uptrend particularly well - a flag breaking out
    near its 52-week high, in a stock already outperforming the broader
    market, is the classic "strongest stocks in the strongest market"
    setup.

    Unlike every other indicator in this file, this one needs a second
    ticker's data to compare against, so it doesn't fit the plain
    (price_data, pattern) shape the others share. Callers that loop over
    INDICATOR_COMBINATIONS generically (e.g. main.py) need to bind
    benchmark_data in first, e.g. with functools.partial, before calling
    this the same way as the rest.

    price_data: DataFrame with Close column, e.g. the output of
        fetch_daily_price_history() for the ticker being scanned.
    pattern: a TrianglePattern or BullFlagPattern (from src/patterns.py).
        The indicator is evaluated at the pattern's end date.
    benchmark_data: daily OHLCV DataFrame for a benchmark index/sector
        ETF (e.g. SPY), e.g. also from fetch_daily_price_history().
    proximity_window: passed straight through to
        compute_pct_from_52_week_high().
    max_pct_from_high: today's Close has to be within this percentage of
        the 52-week high to count as "near" it. PLAN.md suggests roughly
        10-15%; 15 is used as the more lenient end of that range.
    relative_strength_lookback: passed straight through to
        compute_relative_strength().

    Returns a dict of feature values:
        pct_from_52_week_high: how far today's Close is below its 52-week
            high, as a percentage.
        is_near_52_week_high: True if pct_from_52_week_high is at or
            below max_pct_from_high.
        relative_strength: the ticker's return vs. the benchmark's return
            over relative_strength_lookback bars.
        is_outperforming_benchmark: True if relative_strength is above
            1.0.
    """
    evaluation_date = pattern_evaluation_date(pattern)

    pct_from_high_series = compute_pct_from_52_week_high(price_data, window=proximity_window)
    relative_strength_series = compute_relative_strength(price_data, benchmark_data, lookback=relative_strength_lookback)

    current_pct_from_high = pct_from_high_series.loc[evaluation_date]
    current_relative_strength = relative_strength_series.loc[evaluation_date]

    return {
        "pct_from_52_week_high": current_pct_from_high,
        "is_near_52_week_high": current_pct_from_high <= max_pct_from_high,
        "relative_strength": current_relative_strength,
        "is_outperforming_benchmark": current_relative_strength > 1.0,
    }


# Maps each indicator combination to its feature function, numbered to
# match PLAN.md's Stage 4 table - lets callers (e.g. main.py's
# --indicator argument) pick a combination by number instead of importing
# a specific function by name. Every combination except #5 takes
# (price_data, pattern) and returns a feature dict, so they can all be
# called the same way; #5 additionally needs benchmark_data bound in
# first (see its docstring above).
INDICATOR_COMBINATIONS = {
    1: {
        "name": "Bollinger Band squeeze + volume surge",
        "compute_features": bollinger_squeeze_and_volume_surge,
    },
    2: {
        "name": "ADX/DMI trend filter + MACD histogram flip",
        "compute_features": adx_trend_filter_and_macd_flip,
    },
    3: {
        "name": "Donchian Channel breakout + OBV confirming new high",
        "compute_features": donchian_breakout_and_obv_confirmation,
    },
    4: {
        "name": "RSI momentum shift + ATR expansion off a low",
        "compute_features": rsi_momentum_shift_and_atr_expansion,
    },
    5: {
        "name": "Proximity to 52-week high + relative strength vs. benchmark",
        "compute_features": near_52_week_high_and_relative_strength,
        # Flags that this combination needs a benchmark ticker's data
        # bound in (via functools.partial) before it can be called the
        # same (price_data, pattern) way as every other combination.
        "needs_benchmark": True,
    },
}


if __name__ == "__main__":
    # Quick manual check: fetch AAPL's daily history, detect its patterns,
    # and print every indicator's features for each one, so the numbers
    # can be sanity-checked against the chart by eye.
    import functools

    from scripts.fetch_real_data import fetch_daily_price_history
    from src.patterns import deduplicate_bull_flags, deduplicate_triangles, detect_bull_flags, detect_triangles, find_pivots

    aapl_daily_data = fetch_daily_price_history("AAPL")
    aapl_pivots = find_pivots(aapl_daily_data, order=5)
    aapl_triangles = deduplicate_triangles(detect_triangles(aapl_pivots))
    aapl_bull_flags = deduplicate_bull_flags(detect_bull_flags(aapl_daily_data))

    aapl_patterns = aapl_triangles + aapl_bull_flags

    # Combination #5 needs a benchmark ticker's data too - fetch it once
    # and bind it in with functools.partial, so it can be called the same
    # (price_data, pattern) way as every other combination below.
    spy_daily_data = fetch_daily_price_history("SPY")

    for indicator_number, combination in INDICATOR_COMBINATIONS.items():
        print(f"--- Indicator #{indicator_number}: {combination['name']} ---")
        compute_features = combination["compute_features"]
        if combination.get("needs_benchmark"):
            compute_features = functools.partial(compute_features, benchmark_data=spy_daily_data)

        for pattern in aapl_patterns:
            features = compute_features(aapl_daily_data, pattern)
            if isinstance(pattern, TrianglePattern):
                print(f"{pattern.triangle_type} triangle ending {pattern.end_date.date()}: {features}")
            elif isinstance(pattern, BullFlagPattern):
                print(f"bull flag ending {pattern.flag_end_date.date()}: {features}")
