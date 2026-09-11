"""
Interactive TradingView-style dark-themed candlestick + volume chart for
daily OHLCV price data, built with Plotly.

This is the single plotting entry point described in PLAN.md (Stage 5b):
as later stages (pivots, pattern detection, confirmation indicators) produce
output, they're passed into plot_chart() as extra overlays/panels, so the
chart looks the same everywhere it's used instead of every script building
its own plot.

Renders to a self-contained HTML file and opens it in the default browser,
rather than a static image. This gives native hover tooltips (candle OHLC,
pivot/pattern values, indicator readings) and pan/zoom for free, and was
chosen over the project's original matplotlib/mplfinance charting because
matplotlib's interactive window didn't reliably render scatter overlays on
this machine - see PLAN.md's Stage 5b notes for the full reasoning behind
the switch.
"""

import os
import tempfile

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.patterns import BullFlagPattern, TrianglePattern

# Colors matching the project's original TradingView-inspired dark theme.
_BACKGROUND_COLOR = "#131722"
_GRID_COLOR = "#2a2e39"
_TEXT_COLOR = "#d1d4dc"
_UP_COLOR = "#26a69a"
_DOWN_COLOR = "#ef5350"

# Fallback colors for extra_panels lines that don't specify their own
# color, cycled through in the order the lines are given.
_DEFAULT_PANEL_LINE_COLORS = ["#42a5f5", "#ffa726", "#ab47bc", "#66bb6a", "#ef5350"]

# A dashed vertical crosshair line at the hovered date, spanning every
# panel at once. Plotly's built-in per-axis spike lines
# (showspikes/spikemode="across") only extend across the one subplot
# they're set on, not through independently-numbered subplot axes like
# the ones make_subplots creates here - so instead, this draws the line
# by hand: on every hover, it adds a shape anchored to the shared x-axis
# ("xref": "x") but spanning the full figure height ("yref": "paper",
# 0 to 1), which correctly reaches through every stacked panel; on
# unhover, the shape is removed.
_CROSSHAIR_SCRIPT = f"""
var plotDiv = document.getElementsByClassName('plotly-graph-div')[0];
plotDiv.on('plotly_hover', function(eventData) {{
    var hoveredX = eventData.points[0].x;
    Plotly.relayout(plotDiv, {{
        shapes: [{{
            type: 'line',
            xref: 'x',
            yref: 'paper',
            x0: hoveredX,
            x1: hoveredX,
            y0: 0,
            y1: 1,
            line: {{ color: '{_TEXT_COLOR}', width: 1, dash: 'dash' }}
        }}]
    }});
}});
plotDiv.on('plotly_unhover', function() {{
    Plotly.relayout(plotDiv, {{ shapes: [] }});
}});
"""


def _triangle_trendline_traces(price_data: pd.DataFrame, triangle: TrianglePattern):
    """
    Build the upper and lower trendline traces for one detected triangle,
    evaluating its fitted line equations at the bar positions of its
    start and end dates.
    """
    start_position = price_data.index.get_loc(triangle.start_date)
    end_position = price_data.index.get_loc(triangle.end_date)
    x_values = [triangle.start_date, triangle.end_date]

    upper_y = [
        triangle.high_slope * start_position + triangle.high_intercept,
        triangle.high_slope * end_position + triangle.high_intercept,
    ]
    lower_y = [
        triangle.low_slope * start_position + triangle.low_intercept,
        triangle.low_slope * end_position + triangle.low_intercept,
    ]

    trendline_style = dict(mode="lines", line=dict(color="#ab47bc", width=1.5), showlegend=False)
    return [
        go.Scatter(x=x_values, y=upper_y, hovertemplate="Triangle high trendline: %{y:.2f}<extra></extra>", **trendline_style),
        go.Scatter(x=x_values, y=lower_y, hovertemplate="Triangle low trendline: %{y:.2f}<extra></extra>", **trendline_style),
    ]


def _bull_flag_line_traces(price_data: pd.DataFrame, bull_flag: BullFlagPattern):
    """
    Build the pole line and flag-range box outline traces for one
    detected bull flag.
    """
    pole_start_price = price_data.loc[bull_flag.pole_start_date, "Close"]
    pole_end_price = price_data.loc[bull_flag.pole_end_date, "Close"]

    flag_style = dict(mode="lines", line=dict(color="#66bb6a", width=1.5), showlegend=False)
    return [
        go.Scatter(
            x=[bull_flag.pole_start_date, bull_flag.pole_end_date],
            y=[pole_start_price, pole_end_price],
            hovertemplate="Bull flag pole: %{y:.2f}<extra></extra>",
            **flag_style,
        ),
        go.Scatter(
            x=[bull_flag.flag_start_date, bull_flag.flag_end_date],
            y=[bull_flag.flag_high, bull_flag.flag_high],
            hovertemplate="Flag high: %{y:.2f}<extra></extra>",
            **flag_style,
        ),
        go.Scatter(
            x=[bull_flag.flag_start_date, bull_flag.flag_end_date],
            y=[bull_flag.flag_low, bull_flag.flag_low],
            hovertemplate="Flag low: %{y:.2f}<extra></extra>",
            **flag_style,
        ),
    ]


# Exit reason -> color, shared between the entry-to-exit line and the
# exit marker, so a label's outcome is visually obvious at a glance:
# green for a target hit, red for a stop-out, grey for timing out.
_LABEL_OUTCOME_COLORS = {"target": "#26a69a", "stop": "#ef5350", "time": "#787b86"}


def _label_traces(label: dict):
    """
    Build the entry-to-exit line and markers for one labeled pattern
    outcome (src/labeling.py's label_pattern_outcome()) - a straight line
    from the entry price/date to the exit price/date, colored by whether
    the exit was a target hit, a stop-out, or a time-based exit.
    """
    outcome_color = _LABEL_OUTCOME_COLORS[label["exit_reason"]]
    trade_line = go.Scatter(
        x=[label["entry_date"], label["exit_date"]],
        y=[label["entry_price"], label["exit_price"]],
        mode="lines+markers",
        line=dict(color=outcome_color, width=2, dash="dot"),
        marker=dict(size=6, color=outcome_color),
        showlegend=False,
        hovertemplate=(
            f"Entry: {label['entry_price']:.2f} on {label['entry_date'].date()}<br>"
            f"Exit ({label['exit_reason']}): {label['exit_price']:.2f} on {label['exit_date'].date()}<br>"
            f"Return: {label['return_pct']:.1f}%<extra></extra>"
        ),
    )
    return [trade_line]


def plot_chart(
    price_data: pd.DataFrame,
    ticker: str = "",
    pivots: pd.DataFrame = None,
    patterns=None,
    price_overlays=None,
    extra_panels=None,
    labels=None,
    save_path: str = None,
):
    """
    Plot an interactive candlestick + volume chart for one stock's daily
    price history. Hovering over any candle, marker, or indicator line
    shows its value; scroll/drag to zoom and pan.

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
    price_overlays: optional list of line specs for indicators that share
        the price panel's own y-scale (e.g. Bollinger Bands, a Donchian
        Channel) rather than needing a separate panel. Each spec is a
        dict:
            {
                "lines": {"Upper Band": series, "Lower Band": series},
                "colors": {"Upper Band": "#787b86", ...},   # optional
            }
        "lines" values are Series with the same index as price_data.
        "colors" is optional per-name overrides; anything not given a
        color there cycles through a default palette. All specs are drawn
        on the same price panel as the candles.
    extra_panels: optional list of panel specs for indicators that need
        their own stacked panel below price/volume, rather than an
        overlay on the price panel (e.g. ADX/DMI, MACD - oscillators with
        their own y-scale unrelated to price). Each spec is a dict:
            {
                "ylabel": "MACD",
                "lines": {"MACD": series, "Signal": series},   # optional
                "bars": {"Histogram": series},                  # optional
                "colors": {"MACD": "#42a5f5", ...},             # optional
            }
        "lines"/"bars" values are Series with the same index as
        price_data. "colors" is optional per-name overrides; anything not
        given a color there cycles through a default palette. One panel
        is added per entry in the list, in order, below the price/volume
        panels.
    labels: optional list of dicts (Stage 5's label_patterns() output) -
        each drawn as a dotted line from the entry to the exit price/date,
        colored green for a target hit, red for a stop-out, or grey for a
        time-based exit.
    save_path: if given, saves the chart permanently to this HTML file
        path (e.g. for building up a folder of chart snapshots). If not
        given, the chart is saved to a temporary HTML file and opened
        automatically in the default browser instead.
    """
    chart_title = f"{ticker} - Daily" if ticker else "Daily Price"

    num_extra_panels = len(extra_panels) if extra_panels else 0
    total_rows = 2 + num_extra_panels

    # The main price panel gets 3x the height of every other panel
    # (volume, and any extra indicator panels), matching the project's
    # original panel proportions.
    row_heights = [3] + [1] * (total_rows - 1)
    row_height_sum = sum(row_heights)
    row_heights = [height / row_height_sum for height in row_heights]

    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data["Open"],
            high=price_data["High"],
            low=price_data["Low"],
            close=price_data["Close"],
            increasing_line_color=_UP_COLOR,
            decreasing_line_color=_DOWN_COLOR,
            increasing_fillcolor=_UP_COLOR,
            decreasing_fillcolor=_DOWN_COLOR,
            name="Price",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    if pivots is not None:
        # NaN values (every non-pivot bar) are automatically skipped by
        # Plotly, so only the actual pivot bars get a marker drawn.
        fig.add_trace(
            go.Scatter(
                x=pivots.index,
                y=pivots["swing_high"],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color="#ffa726", line=dict(color="black", width=1)),
                name="Swing High",
                showlegend=False,
                hovertemplate="Swing High: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=pivots.index,
                y=pivots["swing_low"],
                mode="markers",
                marker=dict(symbol="triangle-up", size=12, color="#42a5f5", line=dict(color="black", width=1)),
                name="Swing Low",
                showlegend=False,
                hovertemplate="Swing Low: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    for overlay_spec in price_overlays or []:
        overlay_colors = overlay_spec.get("colors", {})
        next_default_color = 0
        for name, series in overlay_spec.get("lines", {}).items():
            color = overlay_colors.get(name)
            if color is None:
                color = _DEFAULT_PANEL_LINE_COLORS[next_default_color % len(_DEFAULT_PANEL_LINE_COLORS)]
                next_default_color += 1
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    mode="lines",
                    line=dict(color=color, width=1),
                    name=name,
                    showlegend=False,
                    hovertemplate=f"{name}: " + "%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if patterns:
        # Triangle trendlines are drawn in purple, bull flag pole/box
        # lines in green, so the two pattern types stay visually distinct
        # from each other and from the orange/blue pivot markers above.
        for pattern in patterns:
            if isinstance(pattern, TrianglePattern):
                for trace in _triangle_trendline_traces(price_data, pattern):
                    fig.add_trace(trace, row=1, col=1)
            elif isinstance(pattern, BullFlagPattern):
                for trace in _bull_flag_line_traces(price_data, pattern):
                    fig.add_trace(trace, row=1, col=1)

    if labels:
        for label in labels:
            for trace in _label_traces(label):
                fig.add_trace(trace, row=1, col=1)

    # Volume panel: color each bar the same up/down color as its candle.
    volume_colors = [
        _UP_COLOR if close_price >= open_price else _DOWN_COLOR
        for open_price, close_price in zip(price_data["Open"], price_data["Close"])
    ]
    fig.add_trace(
        go.Bar(x=price_data.index, y=price_data["Volume"], marker_color=volume_colors, name="Volume", showlegend=False),
        row=2,
        col=1,
    )

    # Extra indicator panels: each spec's lines/bars get their own row,
    # stacked below price/volume in the order given.
    for panel_offset, panel_spec in enumerate(extra_panels or []):
        panel_row = 3 + panel_offset
        panel_colors = panel_spec.get("colors", {})
        next_default_color = 0

        for name, series in panel_spec.get("lines", {}).items():
            color = panel_colors.get(name)
            if color is None:
                color = _DEFAULT_PANEL_LINE_COLORS[next_default_color % len(_DEFAULT_PANEL_LINE_COLORS)]
                next_default_color += 1
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    mode="lines",
                    line=dict(color=color, width=1.2),
                    name=name,
                    showlegend=False,
                    hovertemplate=f"{name}: " + "%{y:.2f}<extra></extra>",
                ),
                row=panel_row,
                col=1,
            )

        for name, series in panel_spec.get("bars", {}).items():
            color = panel_colors.get(name, "#787b86")
            fig.add_trace(
                go.Bar(x=series.index, y=series, marker_color=color, opacity=0.6, name=name, showlegend=False),
                row=panel_row,
                col=1,
            )

        fig.update_yaxes(title_text=panel_spec.get("ylabel", ""), row=panel_row, col=1)

    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    # Candlestick traces add their own range slider below the chart by
    # default; it's redundant with normal scroll/drag zooming, so switch
    # it off.
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)

    # Skip weekends so trading days stay packed together with no empty
    # gaps (matching the project's original TradingView-style look), and
    # show one x-axis label per calendar month.
    fig.update_xaxes(
        rangebreaks=[dict(bounds=["sat", "mon"])],
        dtick="M1",
        tickformat="%Y %b",
        tickangle=45,
        gridcolor=_GRID_COLOR,
    )
    fig.update_yaxes(gridcolor=_GRID_COLOR)

    fig.update_layout(
        title=chart_title,
        template="plotly_dark",
        paper_bgcolor=_BACKGROUND_COLOR,
        plot_bgcolor=_BACKGROUND_COLOR,
        font=dict(color=_TEXT_COLOR),
        # "x unified" shows every panel's value at the hovered date
        # together in one tooltip - e.g. price, volume, and ADX/MACD all
        # at once - closer to TradingView's crosshair info panel than
        # Plotly's default one-trace-at-a-time hover.
        hovermode="x unified",
        height=250 * total_rows + 150,
        margin=dict(t=60, b=80),
    )

    if save_path:
        fig.write_html(save_path, include_plotlyjs=True, post_script=_CROSSHAIR_SCRIPT)
    else:
        # No permanent save path given: render to a temporary HTML file
        # and open it in the default browser. include_plotlyjs=True embeds
        # the whole Plotly.js library in the file instead of loading it
        # from a CDN, so the chart still renders correctly without an
        # internet connection.
        temp_file = tempfile.NamedTemporaryFile(suffix=".html", delete=False)
        temp_file.close()
        fig.write_html(temp_file.name, include_plotlyjs=True, post_script=_CROSSHAIR_SCRIPT)
        os.startfile(temp_file.name)


if __name__ == "__main__":
    # Quick manual check: fetch AAPL's daily history and plot it, so we can
    # see what the raw data looks like before adding pivots/patterns on
    # top of it in later stages.
    from scripts.fetch_real_data import fetch_daily_price_history

    aapl_daily_data = fetch_daily_price_history("AAPL")
    plot_chart(aapl_daily_data, ticker="AAPL")
