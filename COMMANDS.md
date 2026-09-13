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
list), `lxml` (parsing that list's HTML table), `scikit-learn` (Stage 6's
model - logistic regression, standardization, walk-forward splitting).

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
  -5% look identical by win rate alone, but not by return. Entry is
  triggered by the pattern's real breakout day (`find_breakout_date()`),
  not its detection window's own end date - a triangle whose real move
  was a breakdown has no valid long entry and won't appear in the label
  output at all; pass `entry_trigger="end_date"` directly to
  `label_patterns()`/`label_pattern_outcome()` (not exposed as a `main.py`
  flag) to get the original, superseded rule back.

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

### `scripts/build_breakout_dataset.py` — Stage 6's training dataset, read at the real breakout day

The training dataset Stage 6's model actually uses. It's `build_dataset.py`
with one change: every indicator combination's features are computed at
each pattern's real breakout day (found via `find_breakout_date()`)
instead of at `end_date`/`flag_end_date` - motivated by
`measure_indicator_lag.py`'s finding that reading indicators off the
detection window's own end date, rather than the real breakout, flips the
confirmation-count/return relationship's sign. The label itself
(`entry_date`, `return_pct`, `is_successful` - Stage 5's exit rule) is
completely unchanged; only which day the *features* come from is
different, so this is a fair comparison against `labeled_patterns.csv`
rather than a different trading strategy.

```
python -m scripts.build_breakout_dataset                          # full S&P 500, ~2y each, saves to data/breakout_labeled_patterns.csv
python -m scripts.build_breakout_dataset --period 5y                # more history per ticker, more (and more varied) patterns to train on
python -m scripts.build_breakout_dataset --limit 20                # quick test run on the first 20 tickers only
python -m scripts.build_breakout_dataset --max-search-days 10      # give up looking for a breakout sooner
python -m scripts.build_breakout_dataset --output data/other.csv   # save somewhere else
```

Patterns with no real breakout found within `--max-search-days` (about 5%
of all patterns) are dropped entirely from this dataset, rather than
falling back to `end_date` for just those rows - mixing two different
evaluation rules into one dataset would reintroduce the exact timing
problem this script exists to avoid.

Its feature-timing search uses the exact same `direction="up",
require_full_candle=True` settings as `label_pattern_outcome()`'s own
entry trigger, so a row's features are always read as of the same day
its trade actually enters on - an earlier version used a looser,
different search for features than for entry, which disagreed on 22% of
rows (see `PLAN.md`'s Stage 6 notes).

Also attaches, per row: Stage 3's own geometric pattern-quality measures
(`pattern_high_r_squared`/`pattern_low_r_squared`/`pattern_contraction_pct`
for triangles, `pattern_pole_return_pct`/`pattern_flag_volume_ratio`/
`pattern_retracement_pct` for bull flags - 0 for whichever three don't
apply to that row's `pattern_type`), and two market-regime descriptors
from the benchmark ticker's own price series
(`market_pct_from_50d_average`, `market_20d_volatility_pct` - see
`_compute_market_regime_series()`). The pattern-quality
measures were computed all along, inside `detect_triangles()`/
`detect_bull_flags()`, but never reached the model until now; the
market-regime descriptors are new. See `PLAN.md`'s Stage 6 notes.

### `src/model.py` — Stage 6's models

Two classifiers predicting whether a pattern's breakout will hit its
target before its stop, trained on `data/breakout_labeled_patterns.csv`'s
features and compared side by side: a logistic regression (picked first
because its learned weights can be read directly, rather than staying a
black box) and a gradient boosting classifier (picked second as a more
flexible model that can pick up on interactions between features). Two
regressors - the magnitude-aware counterparts, predicting `return_pct`
directly instead of the binary target - are also trained and compared.
All four run through the same walk-forward validation - 5 chronological
folds (`sklearn.TimeSeriesSplit`) - each fold tested only on trades that
happen after everything its own training data covers, never a random
split.

```
python -m src.model
```

Runs both classifiers on two feature sets in turn - `FEATURE_COLUMNS`
(every feature) and `CONTINUOUS_FEATURE_COLUMNS` (just the continuous/
ratio readings, with all 12 True/False confirmation flags left out) - to
test whether those flags add anything beyond the raw numbers underneath
them (see `PLAN.md`'s Stage 6 notes: on the original feature set, they
didn't). Both classifiers are wrapped in `GridSearchCV`, searching
hyperparameters *inside* each outer walk-forward fold's own training
data (nested, chronological tuning - see `build_logistic_regression_model()`/
`build_gradient_boosting_model()`'s docstrings for why nesting matters).
Prints, per model, per fold: how many patterns were in the training/test
split, the test fold's actual win rate (the baseline to beat), the win
rate among patterns the model called "will succeed," the win rate among
just the 20% of patterns it was most confident about, and the standard
accuracy/precision/recall/ROC-AUC metrics - then a summary table of mean
ROC-AUC by feature set and model, each model's own view of which
features mattered (fit on the full feature set): logistic regression's
learned weight per feature and gradient boosting's feature importances
(see `PLAN.md`'s Stage 6 notes for why these two views disagree sharply
on which features matter) - and finally the two regressors' walk-forward
results (baseline mean return, mean return among the top 20% predicted,
and the correlation between predicted and actual return per fold).

Two columns (`indicator5_relative_strength`,
`market_pct_from_50d_average`) are imputed with a neutral fill value
(`IMPUTED_FEATURE_FILL_VALUES`) rather than dropped wherever they're
missing only because a pattern's breakout landed too early in its own
fetched history for that column's lookback to be complete - any other
missing feature still causes that row to be dropped, as the simplest
first pass. See `PLAN.md`'s Stage 6 notes for the exact counts.

---

## What each piece does

| File | Function | Stage | What it does |
|---|---|---|---|
| `scripts/fetch_real_data.py` | `fetch_daily_price_history(ticker, period="2y", use_cache=True)` | 1 | Pulls daily OHLCV data for a ticker via `yfinance`, caching to `data/` so repeat calls don't re-download. |
| `scripts/fetch_real_data.py` | `fetch_sp500_tickers()` | 1 | Scrapes the current S&P 500 constituent list from Wikipedia (today's membership, not point-in-time history). |
| `src/patterns.py` | `find_pivots(price_data, order=5)` | 2 | Finds swing highs/lows using a fractal (rolling-extrema) method. |
| `src/patterns.py` | `detect_triangles(pivots, window=40, ...)` | 3 | Fits trendlines to swing highs/lows in a sliding window, classifies converging shapes as symmetrical/ascending/descending. Returns every overlapping window candidate, unfiltered. |
| `src/patterns.py` | `deduplicate_triangles(triangles)` | 3 (chart helper) | Collapses overlapping triangle candidates down to the single best-fitting one per cluster, purely to keep charts readable. |
| `src/patterns.py` | `detect_bull_flags(price_data, pole_lookback=10, ...)` | 3 | Finds a sharp pole move followed by a tight, low-volume flag consolidation. Returns every overlapping candidate, unfiltered. `BullFlagPattern.retracement_pct` (how much of the pole's gain the flag gave back) is stored on the result now, not just used internally to filter. |
| `src/patterns.py` | `deduplicate_bull_flags(bull_flags)` | 3 (chart helper) | Collapses overlapping bull-flag candidates down to the one with the strongest pole per cluster, purely to keep charts and datasets clean. |
| `src/patterns.py` | `find_breakout_date(price_data, pattern, max_search_days=20, direction=None, require_full_candle=False)` | 3 | Finds the first bar where price actually crosses a triangle's trendlines or a bull flag's box, as opposed to `end_date`/`flag_end_date` which is just the detection window's last bar. `direction=None` (default) stops at the first cross in either direction; `direction="up"`/`"down"` searches only that direction, skipping past an opposite-direction cross rather than stopping there. `require_full_candle=True` requires the candle's Low/High (not just its Close) to clear the line - a stricter confirmation. A cross is never reported before the pivots defining that line have actually happened (`TrianglePattern.last_high_pivot_date`/`last_low_pivot_date`), to avoid lookahead bias. Returns `{"breakout_date", "direction"}` or `None`. |
| `src/patterns.py` | `pattern_as_of_breakout(pattern, breakout_date)` | 3 | Returns a copy of a pattern with its evaluation date moved to `breakout_date`, so it plugs into the existing indicator/labeling functions unchanged. |
| `src/indicators.py` | `compute_bollinger_bands` / `compute_adx_dmi` / `compute_macd` / `compute_donchian_channel` / `compute_obv` / `compute_rsi` / `compute_atr` / `compute_pct_from_52_week_high` / `compute_relative_strength` | 4 | The full-series version of each building block, for charting (each corresponding `*_and_*`/`*_confirmation` feature function below evaluates one of these at a single pattern's end date instead). |
| `src/indicators.py` | `bollinger_squeeze_and_volume_surge(price_data, pattern, ...)` | 4 | Indicator #1: Bollinger Band squeeze (narrow bands vs. their own recent history) + volume surge. |
| `src/indicators.py` | `adx_trend_filter_and_macd_flip(price_data, pattern, ...)` | 4 | Indicator #2: ADX/DMI trend filter (real, upward trend) + a recent bullish MACD histogram flip. |
| `src/indicators.py` | `donchian_breakout_and_obv_confirmation(price_data, pattern, ...)` | 4 | Indicator #3: a Donchian Channel breakout + OBV also making a new high (volume confirming the move). |
| `src/indicators.py` | `rsi_momentum_shift_and_atr_expansion(price_data, pattern, ...)` | 4 | Indicator #4: RSI shifting from a "basing" reading up through a momentum threshold + ATR expanding off a recent low (VCP-style). |
| `src/indicators.py` | `near_52_week_high_and_relative_strength(price_data, pattern, benchmark_data, ...)` | 4 | Indicator #5: Close within X% of its 52-week high + outperforming a benchmark ticker's return. Needs a second ticker's data (`benchmark_data`) - see its docstring for how callers bind that in with `functools.partial`. |
| `src/indicators.py` | `INDICATOR_COMBINATIONS` | 4 | Dict mapping each indicator's PLAN.md number (1-5, all built) to its name and feature function - how `main.py`'s `--indicator` looks up which one to run. |
| `src/labeling.py` | `label_pattern_outcome(price_data, pattern, target_pct=10.0, stop_pct=5.0, max_holding_days=20, entry_trigger="breakout", breakout_search_days=20)` | 5 | Labels one pattern's breakout outcome using a fixed target/stop/time exit rule, entering at the Open right after the pattern's real breakout day (`entry_trigger="breakout"`, the default - requires the whole candle, not just its Close, to clear the trendline) or right after its `end_date`/`flag_end_date` (`entry_trigger="end_date"`, superseded). Returns `None` if there's no valid entry trigger, or no full bar of data after it yet. |
| `src/labeling.py` | `label_patterns(price_data, patterns, ...)` | 5 | Runs `label_pattern_outcome()` over a list of patterns, skipping ones that return `None`, and tags each result with its source pattern/type. |
| `src/charting.py` | `plot_chart(price_data, ticker="", pivots=None, patterns=None, price_overlays=None, extra_panels=None, labels=None, breakout_markers=None, save_path=None)` | 5b | Renders an interactive Plotly candlestick + volume chart, one x-axis label per calendar month, with a dashed vertical crosshair on hover spanning every panel. Draws pivot markers if `pivots` is given, triangle/bull-flag overlays if `patterns` is given, price-scale indicator lines if `price_overlays` is given, stacked indicator panels if `extra_panels` is given, labeled trade lines if `labels` is given, and amber star markers at each real breakout day (see `find_breakout_date()`) if `breakout_markers` is given. |
| `src/patterns.py` | `_SUPERSEDED_MIN_PIVOTS_PER_SIDE` | 3 | Documents the history behind `detect_triangles()`'s `min_pivots_per_side` (2, both sides) and `min_pivots_one_side` (3, at least one side) thresholds - including a stricter version (3 on both sides) that was tried and reverted for being too aggressive - kept as a comment so reverting either choice doesn't require digging through git history. See `PLAN.md`'s Stage 3 notes. |
| `main.py` | `run(ticker, pivot_order=5, period="2y", indicator_number=None, label_outcomes=False, save_path=None)` | — | Chains all of the above into one end-to-end run: fetch → pivots → triangles/bull-flags → (optionally) indicator features → (optionally) labeling → plot. `save_path` is forwarded to `plot_chart()`, mainly for scripted callers like `scripts/end_to_end_test.py`. |
| `scripts/end_to_end_test.py` | `run_smoke_test(ticker="AAPL", benchmark_ticker="SPY")` | — | Runs the full pipeline through every indicator combination and labeling, and reports PASS/FAIL per check. See above. |
| `scripts/build_dataset.py` | `build_labeled_dataset(tickers, period="2y", pivot_order=5, benchmark_ticker="SPY")` | 6 | Runs detection + labeling + all 5 indicator combinations across a list of tickers and combines every labeled pattern (with features) into one DataFrame, skipping tickers that fail to fetch. See above. |
| `scripts/measure_indicator_lag.py` | `measure_indicator_lag(tickers, period="2y", benchmark_ticker="SPY", max_search_days=20)` | 3/6 | Recomputes every indicator combination's confirmation flags at both a pattern's `end_date` and its real breakout day, for every labeled pattern across a list of tickers. See above. |
| `scripts/build_breakout_dataset.py` | `build_breakout_dataset(tickers, period="2y", pivot_order=5, benchmark_ticker="SPY", max_search_days=20)` | 6 | Like `build_labeled_dataset()`, but every indicator combination's features are computed at each pattern's real breakout day instead of `end_date`/`flag_end_date`; patterns with no breakout found are dropped. See above. |
| `src/model.py` | `load_training_data(csv_path="data/breakout_labeled_patterns.csv", feature_columns=None)` | 6 | Loads the training dataset, keeps `feature_columns` (defaults to `FEATURE_COLUMNS`; pass `CONTINUOUS_FEATURE_COLUMNS` to drop the 12 confirmation booleans) plus pattern type, imputes two columns (`IMPUTED_FEATURE_FILL_VALUES`) and drops rows still missing a feature, sorts by `entry_date`. Returns `(features, target, entry_dates, return_pct)`. See above. |
| `src/model.py` | `train_and_evaluate(features, target, n_splits=5, build_model=build_logistic_regression_model)` | 6 | Trains/tests whichever classifier `build_model` constructs across 5 chronological walk-forward folds; returns each fold's win-rate and classification metrics. See above. |
| `src/model.py` | `train_and_evaluate_regression(features, return_pct, n_splits=5, build_model=build_linear_regression_model)` | 6 | Same walk-forward idea, but for a regressor predicting `return_pct` directly - returns each fold's baseline/top-20%-predicted mean return and the predicted-vs-actual correlation. See above. |
| `src/model.py` | `build_logistic_regression_model()` / `build_gradient_boosting_model()` | 6 | Each returns a fresh, untrained `GridSearchCV`-wrapped classifier (`class_weight="balanced"` logistic regression tuning `C`, or a `GradientBoostingClassifier` tuning tree depth/learning rate/tree count/leaf size) - the search is nested inside whichever outer walk-forward fold calls `.fit()` on it. |
| `src/model.py` | `build_linear_regression_model()` / `build_gradient_boosting_regressor()` | 6 | The magnitude-aware counterparts - a `StandardScaler` + `LinearRegression` pipeline, or a `GradientBoostingRegressor` - passed into `train_and_evaluate_regression()`. Not hyperparameter-tuned, unlike the classifiers above. |
| `src/model.py` | `print_feature_weights(features, target)` / `print_feature_importances(features, target)` | 6 | Fit one logistic regression / gradient boosting model (via its `GridSearchCV` wrapper's `.best_estimator_`) on all the data and print every feature's learned weight (signed, standardized) or importance (unsigned, sums to 1.0), respectively. |

All threshold values in `detect_triangles`/`detect_bull_flags`, every
indicator combination, and the labeling exit rule are first-pass guesses,
not yet validated against real outcomes — see `PLAN.md`'s Stage 3 "Next"
note for the validation plan.

Stage 6's multi-ticker dataset pipeline is built (`scripts/build_dataset.py`),
but the model itself (and Stage 7's backtesting) is not — see `PLAN.md`
for the full pipeline and current status of each stage.
