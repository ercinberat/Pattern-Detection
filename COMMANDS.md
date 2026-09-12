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
handling), `plotly` (interactive charting), `scipy` (trendline regression
fits for triangle detection), `requests` (fetching the S&P 500 ticker
list), `lxml` (parsing that list's HTML table).

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
python main.py AAPL --indicator 5      # also run 52-week-high proximity + relative strength vs. SPY
python main.py AAPL --label            # also label each pattern's outcome (Stage 5) and draw it on the chart
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
  combination to run, numbered per `PLAN.md`'s table:
  - `1` = Bollinger Band squeeze + volume surge
  - `2` = ADX/DMI trend filter + MACD histogram flip
  - `3` = Donchian Channel breakout + OBV confirming new high
  - `4` = RSI momentum shift + ATR expansion off a low
  - `5` = Proximity to 52-week high + relative strength vs. a benchmark
    (SPY by default) - this one fetches a second ticker's data too, so it
    takes a little longer than the others.

  If omitted, Stage 4 is skipped entirely - no console output, no extra
  chart content.
- `--label` (optional flag, default off) — label each detected pattern's
  outcome using Stage 5's fixed target/stop/time exit rule (10% target,
  5% stop, 20-bar max hold, by default), print a win-rate + mean-return
  summary, and draw each labeled trade on the chart. Win rate only counts
  an exit_reason of "target" as a win, so mean return is reported
  alongside it - a pattern that times out at +7% and one stopped out at
  -5% look identical by win rate alone, but not by return.

The chart is an interactive Plotly page that opens automatically in your
default browser (not a static image) - hover over any candle, marker, or
indicator line to see its exact value, and scroll/drag to zoom and pan.
Plotly was chosen over the project's original matplotlib/mplfinance
charting because matplotlib's interactive window didn't reliably render
scatter overlays on this machine; see `PLAN.md`'s Stage 5b notes for the
full reasoning.

Triangle trendlines are drawn in purple, bull flag pole/box lines in
green, on top of the orange/blue pivot markers - hovering over a
trendline shows its fit quality (r²) and contraction %, and hovering
over a flag line shows the pole's return % and the flag's volume ratio.
Indicators #1, #3, and #5
add lines directly on the price panel (Bollinger Bands, Donchian Channel,
52-week high); #2, #3, #4, and #5 also add their own panel(s) stacked
below price/volume (ADX/DMI + MACD, OBV, RSI + ATR, Relative Strength).
The x-axis shows one label per calendar month (e.g. "2025 Jul").

Hovering anywhere on the chart draws a dashed vertical crosshair line
through every panel at that date, not just the one panel being hovered -
useful for reading price/volume/indicator values together at a glance.

When `--label` is given, each labeled trade is drawn as a dotted line
from its entry to its exit price/date: green if the target was hit,
red if stopped out, grey if it timed out without hitting either.

### Running individual stages directly

Each module also has its own quick manual check under `if __name__ ==
"__main__":`, useful for testing that one stage in isolation without going
through the full pipeline:

```
python -m scripts.fetch_real_data   # fetches AAPL data, prints it, and prints the current S&P 500 ticker list (no chart)
python -m src.charting              # fetches AAPL data, plots plain candlestick + volume chart
python -m src.patterns              # fetches AAPL data, finds pivots, plots chart with pivot markers
python -m src.indicators            # fetches AAPL data, detects patterns, prints every indicator's features (no chart)
python -m src.labeling              # fetches AAPL data, detects patterns, prints each one's labeled outcome (no chart)
```

Note: run these with `-m` (e.g. `python -m src.patterns`), not
`python src/patterns.py` — plain script invocation doesn't put the repo
root on Python's import path, so the cross-module imports (e.g.
`src.patterns` importing from `scripts.fetch_real_data`) fail with
`ModuleNotFoundError`.

### `scripts/end_to_end_test.py` — smoke test

Runs the full real pipeline for a ticker (default AAPL) through every
stage - pattern detection, every Stage 4 indicator combination, Stage 5
labeling, and chart generation (with and without indicators/labels) -
and prints PASS/FAIL for each step - catches exceptions, NaN values, and
empty pattern detection. Not a pytest suite (it hits the live network via
yfinance for real price data), so run it by hand after making changes
rather than as part of an automated test run:

```
python -m scripts.end_to_end_test
```

Exits with code 0 if everything passed, 1 if anything failed - the
failing check's label and, where relevant, the exception message are
printed above the final summary line.

### `scripts/build_dataset.py` — Stage 6's training dataset

Runs pivot/pattern detection, labeling, and all five Stage 4 indicator
combinations across a whole universe of tickers (the current S&P 500
constituents by default) instead of just one, and saves every labeled
pattern - features and all - to a single CSV: the multi-ticker training
set Stage 6's model (and Stage 3's threshold-validation sweep) needs,
since one ticker's history alone isn't enough examples for either to mean
anything statistically. Each indicator combination's columns are
prefixed `indicator{N}_` (e.g. `indicator1_is_squeezed`) so all five sit
side by side without colliding. A few `indicator5_relative_strength`
values come back blank for patterns too early in the fetched window to
have 63 prior trading days of history - expected, not a bug.

```
python -m scripts.build_dataset                          # full S&P 500, ~2y each, saves to data/labeled_patterns.csv
python -m scripts.build_dataset --limit 20                # quick test run on the first 20 tickers only
python -m scripts.build_dataset --period 5y               # more history per ticker
python -m scripts.build_dataset --output data/other.csv   # save somewhere else
```

Tickers that fail to fetch (delisted, too new to have a full `period` of
history, etc.) are skipped with a printed message rather than stopping
the whole run. Scanning the full universe makes ~500 network calls, so
it takes a while - each ticker's data gets cached (see
`fetch_daily_price_history`'s `use_cache`), so re-running the script
later doesn't re-download tickers it already has.

### `scripts/measure_indicator_lag.py` — indicator-timing measurement

Tests whether reading Stage 4's indicators at a pattern's own
`end_date`/`flag_end_date` (the last bar of its detection window) instead
of at the real breakout/breakdown candle makes a difference - built after
a real case (EXPE, see `PLAN.md`'s Stage 3 notes) showed the two can land
days apart. For every labeled pattern across a universe of tickers, it
finds the real breakout day (`find_breakout_date()`) and recomputes every
registered indicator combination's confirmation flags at both dates, so
`confirmation_count`'s relationship with `return_pct` can be compared
under each timing. This is a standing measure, not a one-off script - any
indicator combination added to `INDICATOR_COMBINATIONS` in the future
gets the same before/after comparison automatically (aside from listing
its boolean feature names in this script's `_BOOLEAN_FEATURE_NAMES`).

```
python -m scripts.measure_indicator_lag                              # full S&P 500, ~2y each
python -m scripts.measure_indicator_lag --limit 20                   # quick test run on the first 20 tickers only
python -m scripts.measure_indicator_lag --max-search-days 10         # give up looking for a breakout sooner
python -m scripts.measure_indicator_lag --output data/other.csv      # save somewhere else
```

Patterns whose geometry never gets broken within `--max-search-days` (default
20, matching Stage 5's `max_holding_days`) are counted but excluded from
the comparison, since there's no "at breakout" reading to compare against.

---

## What each piece does

| File | Function | Stage | What it does |
|---|---|---|---|
| `scripts/fetch_real_data.py` | `fetch_daily_price_history(ticker, period="2y", use_cache=True)` | 1 | Pulls daily OHLCV data for a ticker via `yfinance`, caching to `data/` so repeat calls don't re-download. |
| `scripts/fetch_real_data.py` | `fetch_sp500_tickers()` | 1 | Scrapes the current S&P 500 constituent list from Wikipedia (today's membership, not point-in-time history). |
| `src/patterns.py` | `find_pivots(price_data, order=5)` | 2 | Finds swing highs/lows using a fractal (rolling-extrema) method. |
| `src/patterns.py` | `detect_triangles(pivots, window=40, ...)` | 3 | Fits trendlines to swing highs/lows in a sliding window, classifies converging shapes as symmetrical/ascending/descending. Returns every overlapping window candidate, unfiltered. |
| `src/patterns.py` | `deduplicate_triangles(triangles)` | 3 (chart helper) | Collapses overlapping triangle candidates down to the single best-fitting one per cluster, purely to keep charts readable. |
| `src/patterns.py` | `detect_bull_flags(price_data, pole_lookback=10, ...)` | 3 | Finds a sharp pole move followed by a tight, low-volume flag consolidation. Returns every overlapping candidate, unfiltered. |
| `src/patterns.py` | `deduplicate_bull_flags(bull_flags)` | 3 (chart helper) | Collapses overlapping bull-flag candidates down to the one with the strongest pole per cluster, purely to keep charts and datasets clean. |
| `src/patterns.py` | `find_breakout_date(price_data, pattern, max_search_days=20)` | 3 | Finds the first bar where price actually crosses a triangle's trendlines or a bull flag's box, as opposed to `end_date`/`flag_end_date` which is just the detection window's last bar. Returns `{"breakout_date", "direction"}` or `None`. |
| `src/patterns.py` | `pattern_as_of_breakout(pattern, breakout_date)` | 3 | Returns a copy of a pattern with its evaluation date moved to `breakout_date`, so it plugs into the existing indicator/labeling functions unchanged. |
| `src/indicators.py` | `compute_bollinger_bands` / `compute_adx_dmi` / `compute_macd` / `compute_donchian_channel` / `compute_obv` / `compute_rsi` / `compute_atr` / `compute_pct_from_52_week_high` / `compute_relative_strength` | 4 | The full-series version of each building block, for charting (each corresponding `*_and_*`/`*_confirmation` feature function below evaluates one of these at a single pattern's end date instead). |
| `src/indicators.py` | `bollinger_squeeze_and_volume_surge(price_data, pattern, ...)` | 4 | Indicator #1: Bollinger Band squeeze (narrow bands vs. their own recent history) + volume surge. |
| `src/indicators.py` | `adx_trend_filter_and_macd_flip(price_data, pattern, ...)` | 4 | Indicator #2: ADX/DMI trend filter (real, upward trend) + a recent bullish MACD histogram flip. |
| `src/indicators.py` | `donchian_breakout_and_obv_confirmation(price_data, pattern, ...)` | 4 | Indicator #3: a Donchian Channel breakout + OBV also making a new high (volume confirming the move). |
| `src/indicators.py` | `rsi_momentum_shift_and_atr_expansion(price_data, pattern, ...)` | 4 | Indicator #4: RSI shifting from a "basing" reading up through a momentum threshold + ATR expanding off a recent low (VCP-style). |
| `src/indicators.py` | `near_52_week_high_and_relative_strength(price_data, pattern, benchmark_data, ...)` | 4 | Indicator #5: Close within X% of its 52-week high + outperforming a benchmark ticker's return. Needs a second ticker's data (`benchmark_data`) - see its docstring for how callers bind that in with `functools.partial`. |
| `src/indicators.py` | `INDICATOR_COMBINATIONS` | 4 | Dict mapping each indicator's PLAN.md number (1-5, all built) to its name and feature function - how `main.py`'s `--indicator` looks up which one to run. |
| `src/labeling.py` | `label_pattern_outcome(price_data, pattern, target_pct=10.0, stop_pct=5.0, max_holding_days=20)` | 5 | Labels one pattern's breakout outcome using a fixed target/stop/time exit rule, entering at the Open right after the pattern's end date. Returns `None` if there isn't a full bar of data after the pattern yet. |
| `src/labeling.py` | `label_patterns(price_data, patterns, ...)` | 5 | Runs `label_pattern_outcome()` over a list of patterns, skipping ones that return `None`, and tags each result with its source pattern/type. |
| `src/charting.py` | `plot_chart(price_data, ticker="", pivots=None, patterns=None, price_overlays=None, extra_panels=None, labels=None, save_path=None)` | 5b | Renders an interactive Plotly candlestick + volume chart, one x-axis label per calendar month, with a dashed vertical crosshair on hover spanning every panel. Draws pivot markers if `pivots` is given, triangle/bull-flag overlays if `patterns` is given, price-scale indicator lines if `price_overlays` is given, stacked indicator panels if `extra_panels` is given, and labeled trade lines if `labels` is given. |
| `main.py` | `run(ticker, pivot_order=5, period="2y", indicator_number=None, label_outcomes=False, save_path=None)` | — | Chains all of the above into one end-to-end run: fetch → pivots → triangles/bull-flags → (optionally) indicator features → (optionally) labeling → plot. `save_path` is forwarded to `plot_chart()`, mainly for scripted callers like `scripts/end_to_end_test.py`. |
| `scripts/end_to_end_test.py` | `run_smoke_test(ticker="AAPL", benchmark_ticker="SPY")` | — | Runs the full pipeline through every indicator combination and labeling, and reports PASS/FAIL per check. See above. |
| `scripts/build_dataset.py` | `build_labeled_dataset(tickers, period="2y", pivot_order=5, benchmark_ticker="SPY")` | 6 | Runs detection + labeling + all 5 indicator combinations across a list of tickers and combines every labeled pattern (with features) into one DataFrame, skipping tickers that fail to fetch. See above. |
| `scripts/measure_indicator_lag.py` | `measure_indicator_lag(tickers, period="2y", benchmark_ticker="SPY", max_search_days=20)` | 3/6 | Recomputes every indicator combination's confirmation flags at both a pattern's `end_date` and its real breakout day, for every labeled pattern across a list of tickers. See above. |

All threshold values in `detect_triangles`/`detect_bull_flags`, every
indicator combination, and the labeling exit rule are first-pass guesses,
not yet validated against real outcomes — see `PLAN.md`'s Stage 3 "Next"
note for the validation plan.

Stage 6's multi-ticker dataset pipeline is built (`scripts/build_dataset.py`),
but the model itself (and Stage 7's backtesting) is not — see `PLAN.md`
for the full pipeline and current status of each stage.
