"""
TradingView-style dark-themed candlestick + volume chart for daily OHLCV
price data.

This is the single plotting entry point described in PLAN.md (Stage 5b):
as later stages (pivots, pattern detection, confirmation indicators) produce
output, they'll be passed into plot_chart() as extra overlays/panels, so
the chart looks the same everywhere it's used instead of every script
building its own plot.
"""

import os
import tempfile

import mplfinance as mpf
import pandas as pd

from src.patterns import BullFlagPattern, TrianglePattern


# A dark color scheme close to TradingView's default look: dark background,
# teal/red candles for up/down days, and a subtle dashed grid.
_TRADINGVIEW_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=mpf.make_marketcolors(
        up="#26a69a",
        down="#ef5350",
        edge="inherit",
        wick="inherit",
        volume="inherit",
    ),
    facecolor="#131722",
    figcolor="#131722",
    gridcolor="#2a2e39",
    gridstyle="--",
    rc={
        "axes.labelcolor": "#d1d4dc",
        "xtick.color": "#d1d4dc",
        "ytick.color": "#d1d4dc",
        "text.color": "#d1d4dc",
    },
)


def _triangle_trendlines(price_data: pd.DataFrame, triangle: TrianglePattern):
    """
    Build the upper and lower trendline segments for one detected
    triangle, evaluating its fitted line equations at the bar positions
    of its start and end dates.
    """
    start_position = price_data.index.get_loc(triangle.start_date)
    end_position = price_data.index.get_loc(triangle.end_date)

    upper_line = [
        (triangle.start_date, triangle.high_slope * start_position + triangle.high_intercept),
        (triangle.end_date, triangle.high_slope * end_position + triangle.high_intercept),
    ]
    lower_line = [
        (triangle.start_date, triangle.low_slope * start_position + triangle.low_intercept),
        (triangle.end_date, triangle.low_slope * end_position + triangle.low_intercept),
    ]
    return [upper_line, lower_line]


def _bull_flag_lines(price_data: pd.DataFrame, bull_flag: BullFlagPattern):
    """
    Build the pole line (from the pole's start close to its end close)
    and the flag-range box outline (top and bottom of the consolidation)
    for one detected bull flag.
    """
    pole_start_price = price_data.loc[bull_flag.pole_start_date, "Close"]
    pole_end_price = price_data.loc[bull_flag.pole_end_date, "Close"]

    pole_line = [
        (bull_flag.pole_start_date, pole_start_price),
        (bull_flag.pole_end_date, pole_end_price),
    ]
    flag_top = [
        (bull_flag.flag_start_date, bull_flag.flag_high),
        (bull_flag.flag_end_date, bull_flag.flag_high),
    ]
    flag_bottom = [
        (bull_flag.flag_start_date, bull_flag.flag_low),
        (bull_flag.flag_end_date, bull_flag.flag_low),
    ]
    return [pole_line, flag_top, flag_bottom]


def plot_chart(
    price_data: pd.DataFrame,
    ticker: str = "",
    pivots: pd.DataFrame = None,
    patterns=None,
    extra_panels=None,
    save_path: str = None,
):
    """
    Plot a candlestick + volume chart for one stock's daily price history.

    price_data: DataFrame indexed by date with Open/High/Low/Close/Volume
        columns, e.g. the output of fetch_daily_price_history().
    ticker: stock symbol, only used for the chart title.
    pivots: optional output of find_pivots() (Stage 2) - a DataFrame with
        the same index as price_data plus swing_high/swing_low columns.
        When given, swing highs/lows are marked directly on the price
        panel so pivot detection can be sanity-checked visually.
    patterns: optional list of TrianglePattern/BullFlagPattern (Stage 3's
        detect_triangles()/detect_bull_flags() output). Triangle trendlines
        are drawn as two converging lines; bull flags are drawn as a pole
        line plus a box around the flag consolidation.
    extra_panels: reserved for Stage 4 (confirmation indicators). Once
        indicators.py exists, this will hold indicator series (RSI, MACD,
        ADX, etc.) to stack as extra panels below the price panel. Ignored
        for now.
    save_path: if given, saves the chart image permanently to this file
        path (e.g. for building up a folder of chart images). If not
        given, the chart is saved to a temporary image and opened
        automatically in the default image viewer instead - matplotlib's
        own interactive window doesn't reliably render scatter overlays
        (pivot/pattern markers) on every system, but this save-then-open
        approach has been confirmed to render everything correctly.
    """
    chart_title = f"{ticker} - Daily" if ticker else "Daily Price"

    plot_kwargs = dict(
        type="candle",
        style=_TRADINGVIEW_STYLE,
        volume=True,
        title=chart_title,
        ylabel="Price ($)",
        ylabel_lower="Volume",
        figsize=(12, 7),
    )

    if pivots is not None:
        # Draw swing highs as downward triangles and swing lows as upward
        # triangles, right at the pivot bar's High/Low price, using
        # mplfinance's scatter overlay ("addplot"). Each series lines up
        # with price_data bar-for-bar, with NaN on every non-pivot bar, so
        # only the actual pivot bars get a marker drawn.
        pivot_markers = [
            mpf.make_addplot(
                pivots["swing_high"],
                type="scatter",
                marker="v",
                markersize=200,
                color="#ffa726",
                edgecolors="black",
            ),
            mpf.make_addplot(
                pivots["swing_low"],
                type="scatter",
                marker="^",
                markersize=200,
                color="#42a5f5",
                edgecolors="black",
            ),
        ]
        plot_kwargs["addplot"] = pivot_markers

    if patterns:
        # Triangle trendlines are drawn in purple, bull flag pole/box
        # lines in green, so the two pattern types stay visually distinct
        # from each other and from the orange/blue pivot markers above.
        line_segments = []
        line_colors = []
        for pattern in patterns:
            if isinstance(pattern, TrianglePattern):
                segments = _triangle_trendlines(price_data, pattern)
                line_segments.extend(segments)
                line_colors.extend(["#ab47bc"] * len(segments))
            elif isinstance(pattern, BullFlagPattern):
                segments = _bull_flag_lines(price_data, pattern)
                line_segments.extend(segments)
                line_colors.extend(["#66bb6a"] * len(segments))

        if line_segments:
            plot_kwargs["alines"] = dict(alines=line_segments, colors=line_colors, linewidths=[1.5] * len(line_segments))

    if save_path:
        plot_kwargs["savefig"] = save_path
        mpf.plot(price_data, **plot_kwargs)
    else:
        # No permanent save path given: render to a temporary PNG and open
        # it in the default image viewer, so the chart still "pops up"
        # without relying on matplotlib's interactive window.
        temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        temp_image.close()
        plot_kwargs["savefig"] = temp_image.name
        mpf.plot(price_data, **plot_kwargs)
        os.startfile(temp_image.name)


if __name__ == "__main__":
    # Quick manual check: fetch AAPL's daily history and plot it, so we can
    # see what the raw data looks like before adding pivots/patterns on
    # top of it in later stages.
    from scripts.fetch_real_data import fetch_daily_price_history

    aapl_daily_data = fetch_daily_price_history("AAPL")
    plot_chart(aapl_daily_data, ticker="AAPL")
