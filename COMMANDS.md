# COMMANDS.md — How to Run This Project

This file lists the commands for setting up and running the project as it
exists today, and explains what each one does. **Update this file before
every push** so it never falls behind what the code actually does.

---

## Setup

Activate the virtual environment, then install dependencies:

```
.venv\Scripts\activate
pip install -r requirements.txt
```

Current dependencies: `yfinance` (pulling price data), `pandas` (data
handling), `mplfinance` (charting), `scipy` (trendline regression fits for
triangle detection).

---

## Running the pipeline

### `main.py` — the main entry point

Runs the pipeline as it exists today for one ticker: fetch daily price
history, find its swing pivots, detect triangle/bull-flag candidates,
optionally run a Stage 4 confirmation indicator against each one, and
plot the result.

```
python main.py                        # AAPL, 2 years of data, pivot order 5, no indicator
python main.py MSFT                    # a different ticker
python main.py TSLA --order 8          # fewer, more significant swing pivots
python main.py TSLA --period 5y        # more history
python main.py AAPL --indicator 1      # also run Bollinger squeeze + volume surge
```

Arguments:
- `ticker` (optional, default `AAPL`) — stock ticker symbol.
- `--order` (optional, default `5`) — pivot sensitivity. A bar only counts
  as a swing high/low if it beats every other bar within `order` days on
  both sides. Smaller = more, noisier swings; bigger = fewer, more
  significant swings.
- `--period` (optional, default `2y`) — how much daily history to fetch,
  as a yfinance period string (`1y`, `2y`, `5y`, `max`, etc.).
- `--indicator` (optional, default none) — which Stage 4 indicator
  combination to run, numbered per `PLAN.md`'s table (currently only `1`
  exists: Bollinger Band squeeze + volume surge). If omitted, Stage 4 is
  skipped entirely - no console output, no bands on the chart.

The chart opens automatically in your default image viewer. It doesn't use
matplotlib's interactive window — that window doesn't reliably render the
scatter overlays (pivot/pattern markers) on this machine, so `plot_chart()`
always renders to a PNG and opens it instead.

Triangle trendlines are drawn in purple, bull flag pole/box lines in
green, on top of the orange/blue pivot markers. When `--indicator 1` is
given, the Bollinger Bands are drawn as thin grey lines. The x-axis shows
one label per calendar month (e.g. "2025 Jul"), rotated 45 degrees.

### Running individual stages directly

Each module also has its own quick manual check under `if __name__ ==
"__main__":`, useful for testing that one stage in isolation without going
through the full pipeline:

```
python -m scripts.fetch_real_data   # fetches AAPL data and prints it (no chart)
python -m src.charting              # fetches AAPL data, plots plain candlestick + volume chart
python -m src.patterns              # fetches AAPL data, finds pivots, plots chart with pivot markers
```

Note: run these with `-m` (e.g. `python -m src.patterns`), not
`python src/patterns.py` — plain script invocation doesn't put the repo
root on Python's import path, so the cross-module imports (e.g.
`src.patterns` importing from `scripts.fetch_real_data`) fail with
`ModuleNotFoundError`.

---

## What each piece does

| File | Function | Stage | What it does |
|---|---|---|---|
| `scripts/fetch_real_data.py` | `fetch_daily_price_history(ticker, period="2y")` | 1 | Pulls daily OHLCV data for a ticker via `yfinance`. |
| `src/patterns.py` | `find_pivots(price_data, order=5)` | 2 | Finds swing highs/lows using a fractal (rolling-extrema) method. |
| `src/patterns.py` | `detect_triangles(pivots, window=40, ...)` | 3 | Fits trendlines to swing highs/lows in a sliding window, classifies converging shapes as symmetrical/ascending/descending. Returns every overlapping window candidate, unfiltered. |
| `src/patterns.py` | `deduplicate_triangles(triangles)` | 3 (chart helper) | Collapses overlapping triangle candidates down to the single best-fitting one per cluster, purely to keep charts readable. |
| `src/patterns.py` | `detect_bull_flags(price_data, pole_lookback=10, ...)` | 3 | Finds a sharp pole move followed by a tight, low-volume flag consolidation. |
| `src/indicators.py` | `compute_bollinger_bands(price_data, band_window=20, band_num_std=2.0)` | 4 | Computes the full Bollinger Band series (middle/upper/lower + width %) for charting. |
| `src/indicators.py` | `bollinger_squeeze_and_volume_surge(price_data, pattern, ...)` | 4 | Indicator combination #1: checks whether a pattern's end date shows a Bollinger Band squeeze (narrow bands vs. their own recent history) and a volume surge. Returns a feature dict. |
| `src/indicators.py` | `INDICATOR_COMBINATIONS` | 4 | Dict mapping each indicator's PLAN.md number to its name and feature function - how `main.py`'s `--indicator` looks up which one to run. Currently only `1` exists. |
| `src/charting.py` | `plot_chart(price_data, ticker="", pivots=None, patterns=None, bollinger_bands=None, extra_panels=None, save_path=None)` | 5b | Renders a TradingView-style dark candlestick + volume chart, one x-axis label per calendar month. Draws pivot markers if `pivots` is given, triangle/bull-flag overlays if `patterns` is given, and Bollinger Bands if `bollinger_bands` is given. `extra_panels` is a placeholder for indicator combinations #2-5, not built yet. |
| `main.py` | `run(ticker, pivot_order=5, period="2y", indicator_number=None)` | — | Chains all of the above into one end-to-end run: fetch → pivots → triangles/bull-flags → (optionally) indicator features → plot. |

All threshold values in `detect_triangles`/`detect_bull_flags` are
first-pass guesses, not yet validated against real outcomes — see
`PLAN.md`'s Stage 3 "Next" note for the validation plan.

Indicator combinations #2-5, Stage 5 (labeling), Stage 6 (modeling), and
Stage 7 (backtesting) are not built yet — see `PLAN.md` for the full
pipeline and current status of each stage.
