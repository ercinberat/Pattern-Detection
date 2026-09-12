# Momentum Pattern Detection — Project Plan

## Goal

Build a tool that scans daily-bar stock data for momentum breakout setups —
triangles and bullish flags — and estimates the probability of a real
breakout (vs. a fakeout), for **swing trading** (holding periods of days to
a few weeks).

The project has three layers:
1. **Rule-based pattern detection** — find geometric candidates (triangle /
   flag shapes) using swing pivots and trendline fits.
2. **Feature-based confirmation / ML scoring** — compute momentum,
   volatility, and volume indicators around each candidate, and use them
   (eventually via a trained model) to separate real breakouts from noise.
3. **Visualization** — render every detection and its supporting indicator
   evidence on a single TradingView-style chart, so any pattern the tool
   flags can be visually sanity-checked at a glance.

---

## Pipeline Stages

### Stage 1 — Data ingestion
- Pull daily OHLCV via `yfinance` for a configurable universe of tickers.
- Cache raw data locally (CSV/parquet) to avoid re-downloading during
  iteration.
- **Status:** done. `fetch_daily_price_history(ticker, period="2y",
  use_cache=True)` pulls daily OHLCV for a single ticker via yfinance,
  caching each ticker's data to a CSV under `data/` (gitignored) so
  re-running a script that fetches the same tickers doesn't re-download
  them. `fetch_sp500_tickers()` scrapes the current S&P 500 constituent
  list from Wikipedia (needs a browser-like `User-Agent` header, or
  Wikipedia returns a 403) - the "configurable universe of tickers" and
  multi-ticker batch support this stage needed, built once Stage 6
  needed many more labeled examples than one ticker could provide (see
  Stage 6 below). Uses today's constituent list, not point-in-time
  historical membership - a stock added to or dropped from the index
  within the fetch window won't necessarily have been a member for all
  of it, a form of survivorship bias worth keeping in mind.

### Stage 2 — Pivot detection
- Identify swing highs/lows using a rolling-extrema (fractal) method.
- Tunable `order` parameter controls pivot significance.
- **Status:** done (`src/patterns.py: find_pivots`). Verified visually via
  `plot_chart()`'s pivot markers on real AAPL data.

### Stage 3 — Geometric pattern detection
- **Triangles:** fit trendlines to swing highs/lows in a sliding window,
  check for range contraction + convergence, classify as
  symmetrical/ascending/descending.
- **Bullish flags:** detect a sharp "pole" move followed by a tight,
  low-volatility, low-volume "flag" consolidation.
- **Status:** done (`src/patterns.py: detect_triangles`,
  `detect_bull_flags`). Both return every overlapping sliding-window
  candidate as-is (by design - Stage 4 is where confirmation/filtering
  belongs); separate `deduplicate_triangles()`/`deduplicate_bull_flags()`
  functions each collapse overlapping candidates down to one per cluster
  (best combined r² for triangles, strongest pole return for bull flags),
  purely for readable charts and clean downstream datasets. The bull-flag
  version was added after `main.py --label` on CIEN showed one real
  pole-and-flag move detected as 12 overlapping candidates - see Stage 6's
  status for the full story. Verified visually on real AAPL/CIEN data via
  `plot_chart()`'s triangle trendline / bull flag pole+box overlays.
- **Next:** all thresholds (r², contraction %, flat-slope %, pole
  return %, flag range/volume/retracement %) are first-pass guesses,
  documented in each function's docstring - they haven't been tuned
  against real outcomes yet. They can't be validated until there's a
  ground-truth outcome to check them against, so the concrete plan is:
  1. Build Stage 5's labeling rule first - it defines what "the pattern
     worked" even means.
  2. Run `detect_triangles`/`detect_bull_flags` across a broad universe of
     real tickers and history (not just AAPL), and label every detected
     pattern's outcome.
  3. Sweep each threshold and check which cutoffs best separate real
     breakouts from fakeouts, using walk-forward splits (per Stage 6's
     validation approach) rather than one lookback period, so the tuned
     values aren't just overfit to a single market regime.
  This only becomes meaningful with enough labeled examples across enough
  tickers/time to see a real pattern in the outcomes - tuning against one
  ticker's last couple of years would just be curve-fitting noise.
- **Known limitation found via live usage (EXPE, descending triangle
  ending 2025-11-10):** a triangle's `end_date` is just
  `window_slice.index[-1]` - the last bar of whichever overlapping 40-day
  window `deduplicate_triangles()` kept - not the bar where price actually
  broke the trendline. On EXPE, the real move was a +17.6% overnight gap
  on 2025-11-07 (an earnings jump, not a genuine triangle breakdown); the
  three overlapping candidate windows ending 11-10/11-11/11-12 all had
  identical trendline fit (combined r² 1.997) because `find_pivots()`
  hadn't yet confirmed any swing pivot from after the gap (it needs 5
  bars either side), so the fit was unchanged across all three end dates
  and `deduplicate_triangles()`'s `max()` tie-break arbitrarily kept the
  earliest one (11-10). Since Stage 5 enters at the Open right after
  `end_date`, this pattern's entry landed on 11-11 - 3 trading days and
  ~4% higher than the actual gap - for reasons unrelated to signal
  quality. This is exactly the kind of case the planned
  "earnings-date proximity" feature (Stage 4's swing-trading-specific
  features, not started yet) and an N-day close-through confirmation
  could help catch or filter out; worth considering when picking which of
  those to build first, and worth keeping in mind that `end_date`/entry
  timing can lag the real trigger by several days, especially right after
  a gap.
  - **Built as a reusable measure, not just a one-off finding:**
    `find_breakout_date(price_data, pattern, max_search_days=20)`
    (`src/patterns.py`) finds the first bar where price actually crosses
    a triangle's own fitted trendlines (extrapolated past the window they
    were fit on) or a bull flag's own box (`flag_high`) - searching the
    triangle's whole detection window as well as the days after
    `end_date`, since (as the EXPE case showed) a real break can fall
    *inside* a window whose `end_date` drifted past it. Returns
    `{"breakout_date", "direction"}` or `None` if nothing breaks within
    the search range. `pattern_as_of_breakout(pattern, breakout_date)`
    hands back a copy of a pattern with its evaluation date moved to that
    real breakout day, so it plugs straight into the existing Stage 4
    indicator functions and Stage 5 labeling without changing either -
    both read a pattern's evaluation date via `pattern_evaluation_date()`.
    Covered by `scripts/end_to_end_test.py`'s new Stage 3+ check.
  - **`scripts/measure_indicator_lag.py` uses both to directly test
    whether Stage 6's Confirmation Paradox is (partly) a measurement-
    timing artifact:** for every labeled pattern, it recomputes all 5
    indicator combinations' confirmation flags both at `end_date` (today's
    pipeline) and at the real breakout day, and compares
    `confirmation_count`'s correlation with `return_pct` under each
    timing. Built to be a standing measure for any current or future
    indicator, not a one-off check - see the script's own docstring.
  - **Full S&P 500 run (`python -m scripts.measure_indicator_lag`)
    confirms the timing hypothesis, at real scale:** 1,734 of 1,833
    labeled patterns (95%) had a real breakout found within 20 days.
    Mean lag was -9.1 days (median 0) between `end_date` and the real
    breakout - on average the "detection window" is already 9 days stale
    by the time it's read, not just occasionally off by a few days as the
    EXPE case suggested. Triangle breakout direction was a near coin flip
    (49% up / 51% down) regardless of a triangle's classified type
    (symmetrical/ascending/descending), confirming that classification
    doesn't predict which way it actually breaks.
    - **`confirmation_count`'s correlation with `return_pct` flips sign**
      depending on when it's measured: **-0.145** at `end_date` (today's
      pipeline - this is what "The Confirmation Paradox" artifact
      reports) vs. **+0.106** measured at the real breakout day. The
      bucketed mean-return table tells the same story more concretely:
      at `end_date`, mean return declines almost monotonically from
      +2.5% (1 signal) to -5.0% (12 signals); at the real breakout day,
      the highest-confirmation buckets (7-11 signals) instead show some
      of the *best* returns (+1.5% to +2.6%), clearly better than the
      lowest buckets (-0.3% to -0.8% at 1-2 signals).
    - **Reading:** a substantial part of the Confirmation Paradox looks
      like a measurement-timing artifact, not proof that more
      confirmation is a bad sign in general - reading the indicators off
      a stale, already-passed detection window makes an otherwise-decent
      signal look backwards. This doesn't fully overturn "The
      Confirmation Paradox" artifact (its numbers, as measured by the
      pipeline that exists today, are correct), but it does mean that
      artifact's framing - and the write-up's suggestion that confirmation
      itself predicts worse outcomes - needs a follow-up caveat or
      revision once discussed, since the more accurate statement is "the
      *pipeline's current timing* makes confirmation look bad, but
      confirmation measured at the right moment looks good." Full
      row-level results in `data/indicator_lag_measurements.csv`
      (gitignored - regenerate with `python -m scripts.measure_indicator_lag`).

### Stage 4 — Confirmation indicators (feature engineering)
For every detected pattern candidate, compute a feature set drawn from five
indicator combinations:

| # | Combination | Best fit |
|---|---|---|
| 1 | Bollinger Band squeeze (20, 2σ width percentile) + volume surge (>1.5–2x 20-day avg) | Triangles |
| 2 | ADX(14)/DMI trend filter + MACD(12,26,9) histogram flip | General gate, both patterns |
| 3 | Donchian Channel(20) breakout + OBV confirming new high | Flag continuations |
| 4 | RSI(14) momentum shift (cross above 55–60 after basing <50) + ATR(14) expansion off a low (VCP-style) | Triangles/flags after a basing period |
| 5 | Proximity to 52-week high (within ~10–15%) + relative strength vs. index/sector ETF | Flags after a strong uptrend |

Plus swing-trading-specific features:
- N-day close-through confirmation (e.g. 2 consecutive closes past the
  trendline/resistance) to filter single-day fakeouts
- Gap-through-pattern flag (gap vs. gradual breakout)
- Earnings-date proximity (if trading single names)

- **Status:** all five indicator combinations built (`src/indicators.py`);
  swing-trading-specific features (N-day close-through confirmation,
  gap-through-pattern flag, earnings-date proximity) not started.
  - Combination #1 (Bollinger Band squeeze + volume surge) is built:
    `compute_bollinger_bands()` returns the full band series for
    charting, and `bollinger_squeeze_and_volume_surge(price_data,
    pattern)` evaluates the squeeze/surge at a given pattern's end date
    and returns a feature dict. Drawn on the chart as an overlay on the
    price panel.
  - Combination #2 (ADX/DMI trend filter + MACD histogram flip) is
    built: `compute_adx_dmi()` (Wilder-smoothed ADX/+DI/-DI) and
    `compute_macd()` (MACD line/signal/histogram), combined in
    `adx_trend_filter_and_macd_flip(price_data, pattern)`. Drawn as two
    stacked panels below price/volume via `charting.py`'s new
    `extra_panels` mechanism (a list of `{ylabel, lines, bars, colors}`
    panel specs), rather than an overlay - ADX/MACD are oscillators with
    their own y-scale, unlike Bollinger Bands.
  - Combination #3 (Donchian Channel breakout + OBV confirming new high)
    is built: `compute_donchian_channel()` (rolling high/low over the
    prior window, excluding today - needed so a breakout is actually
    possible) and `compute_obv()`, combined in
    `donchian_breakout_and_obv_confirmation(price_data, pattern)`. The
    channel shares the price panel's scale (drawn via the new
    `price_overlays` mechanism, generalized from combination #1's
    Bollinger-Band-only overlay so future price-scale indicators don't
    each need their own bespoke `plot_chart()` parameter); OBV gets its
    own `extra_panels` panel since it's on a cumulative-volume scale
    unrelated to price.
  - Combination #4 (RSI momentum shift + ATR expansion off a low) is
    built: `compute_rsi()` and `compute_atr()` (both Wilder-smoothed),
    combined in `rsi_momentum_shift_and_atr_expansion(price_data,
    pattern)`. Both are oscillator/volatility series unrelated to price
    scale, so both get their own `extra_panels` panel.
  - Combination #5 (proximity to 52-week high + relative strength vs. a
    benchmark) is built: `compute_pct_from_52_week_high()` and
    `compute_relative_strength()`, combined in
    `near_52_week_high_and_relative_strength(price_data, pattern,
    benchmark_data)`. This is the one combination that needs a second
    ticker's data (a benchmark index/sector ETF, SPY by default in
    `main.py`) - it doesn't fit the plain `(price_data, pattern)` shape
    the other four share, so `INDICATOR_COMBINATIONS[5]` is flagged
    `needs_benchmark: True` and callers bind `benchmark_data` in with
    `functools.partial` before calling it uniformly with the rest. The
    52-week high is drawn as a `price_overlays` line; relative strength
    gets its own `extra_panels` panel.
  - All five combinations are registered in `INDICATOR_COMBINATIONS` and
    selected via `main.py --indicator N`. The swing-trading-specific
    features (N-day close-through confirmation, gap-through-pattern flag,
    earnings-date proximity) are not built yet.
  - `scripts/end_to_end_test.py` is a smoke test that runs the real
    pipeline through every stage and every indicator combination against
    live data, printing PASS/FAIL per check - not a formal pytest suite
    (see `COMMANDS.md`), but a fast way to catch a broken combination
    after a change without manually re-checking each one by hand.

### Stage 5 — Labeling
- For each detected pattern, label the outcome: did price move > X% within
  N bars after the breakout, without first hitting a stop-loss level?
- Needs careful definition to avoid lookahead bias and to reflect a
  realistic swing-trade exit rule (e.g. target/stop/time-based exit).
- **Status:** done (`src/labeling.py`). `label_pattern_outcome(price_data,
  pattern, target_pct=10.0, stop_pct=5.0, max_holding_days=20)` labels one
  pattern using a fixed target/stop/time exit rule - the option chosen
  from this stage's exit-rule open question below, kept simplest to
  validate against first. Entry is the Open of the bar right after the
  pattern's end date, so the label only ever looks at price data strictly
  after the pattern's own detection point (avoiding lookahead bias). If
  a bar's range covers both the stop and target, the stop is assumed hit
  first (the conservative assumption, since daily bars don't say which
  was actually touched first within the day). `label_patterns()` runs
  this over a list of patterns, skipping ones too close to the end of the
  data to label yet. Wired into `main.py --label`, which prints a
  win-rate summary and draws each labeled trade on the chart (dotted line
  from entry to exit, green/red/grey for target/stop/time).
- ATR-based and trailing-stop exit rules were discussed and deliberately
  left for later (see Open Questions) rather than building all three now.

### Stage 5b — Visualization
- TradingView-style dark theme, multi-panel chart: candlesticks + volume +
  RSI + MACD stacked below price, shared x-axis.
  - **Deviation from this description:** RSI and MACD are not always-on
    default panels. Once Stage 4 grew into five different indicator
    combinations (not just RSI/MACD), every indicator's panel(s) became
    opt-in via `main.py --indicator N` instead - a plain run with no
    `--indicator` shows neither. This is a deliberate call, not an
    oversight: keeping the default chart uncluttered mattered more than
    matching the original always-on RSI/MACD wording once there were five
    combinations to choose from instead of two.
- Every detected pattern is drawn directly on the price panel: triangle
  trendlines (with r² and contraction ratio labeled), bull flag pole/flag
  zones (with pole return and volume ratio labeled), swing high/low pivot
  markers.
- Built as a single `plot_chart()` entry point taking the DataFrame plus
  detected patterns, so output stays visually consistent everywhere it's
  used.
- `patterns` and `extra_panels` arguments take Stage 3's triangle/flag
  detection output and Stage 4's indicator series respectively, without
  the charting code needing to change as new pattern types or
  indicators are added.
- Being built incrementally alongside each stage (rather than only at the
  end) so every stage's output can be visually sanity-checked as it's
  built, per an explicit decision to deviate from strict pipeline order.
- **Status:** done (`src/charting.py: plot_chart`). Renders the
  candlestick + volume chart with Stage 2's pivot markers; Stage 3's
  triangle/bull-flag overlays, now labeled on hover with r²/contraction %
  (triangles) and pole return/volume ratio (bull flags), per this
  stage's original description; all five of Stage 4's indicator
  combinations, each wired to either `price_overlays` (Bollinger Bands,
  Donchian Channel, 52-week high - series sharing the price panel's own
  scale) or `extra_panels` (ADX/DMI, MACD, OBV, RSI, ATR, Relative
  Strength - series needing their own stacked panel); and Stage 5's
  labeled trade outcomes (`labels` parameter - a dotted entry-to-exit
  line per pattern, colored green/red/grey for target/stop/time). A
  dashed vertical crosshair spans every panel on hover.
- **Charting library: Plotly, not matplotlib/mplfinance.** The chart
  originally used mplfinance, but its interactive window didn't reliably
  render scatter overlays (pivot/pattern markers) on this machine —
  candles/volume showed fine, but marker artists silently failed to
  render live even at large sizes, while saving to a file rendered them
  correctly every time. Once genuine interactivity (hover tooltips
  showing exact values, pan/zoom) was wanted on top of that, the project
  switched to Plotly instead of continuing to patch around matplotlib's
  static rendering:
  - Plotly renders self-contained HTML/JS and opens in the browser,
    giving hover tooltips and pan/zoom natively, with no dependency on a
    native GUI toolkit (the exact category of thing that was already
    unreliable here).
    An alternative considered was a Python wrapper around TradingView's
    own `lightweight-charts` JS library, which would give genuine
    drag-resizable panels (Plotly only supports setting panel height
    ratios in code, not resizing them live in the browser) and a more
    authentic TradingView crosshair - but it's a much smaller, newer
    dependency that runs via its own native webview window, reintroducing
    the same class of native-rendering risk this switch was meant to
    avoid. Revisit if drag-resize turns out to matter more than expected.
  - `plot_chart()` always renders to HTML; if no permanent `save_path` is
    given, it saves to a temp file and opens it in the default browser.
  - `include_plotlyjs=True` embeds the whole Plotly.js library in the
    output file (a few MB) rather than loading it from a CDN, so charts
    render correctly offline.

### Stage 6 — Modeling
- Start simple: logistic regression or gradient boosting (XGBoost/LightGBM)
  predicting breakout follow-through probability from Stage 4 features.
- Use walk-forward validation (never a random train/test split on time
  series data) to avoid leaking future information.
- **Status:** the multi-ticker data pipeline this needed is built and has
  been run (`scripts/build_dataset.py`); the model itself is not started
  yet.
  - Scanned all 503 current S&P 500 constituents, 2 years of daily data
    each: 1,833 labeled patterns saved to `data/labeled_patterns.csv`
    (gitignored - regenerate with `python -m scripts.build_dataset`).
    Every ticker fetched successfully; 3 had zero qualifying patterns
    over the window.
  - **Two bugs were found and fixed after the first version of this scan**
    (originally 2,273 patterns), both by real usage rather than by
    inspection:
    1. `fetch_daily_price_history()`'s cache read `pd.read_csv(...,
       parse_dates=True)`, which silently failed to parse the date index
       back into real `Timestamp`s when dates included a timezone offset
       - they came back as plain strings. Anything reading from cache
       worked by coincidental string-equality matching until code called
       a `Timestamp`-only method (`.date()`), which crashed
       `main.py --label` for any ticker being read from cache. Fixed by
       parsing explicitly with `pd.to_datetime(..., utc=True)`.
    2. `detect_bull_flags()` had no deduplication step, unlike
       `detect_triangles()`/`deduplicate_triangles()` - one real
       pole-and-flag move was routinely detected as 5-6 near-duplicate
       bull flags on consecutive pole-start days (seen directly on CIEN's
       chart: 12 patterns collapsed to 5 once fixed). Fixed by adding
       `deduplicate_bull_flags()`, the same clustering approach as
       triangles but keyed on the strongest pole (`pole_return_pct`) per
       cluster, wired in everywhere `detect_bull_flags()` is called.
    Re-running after both fixes changed bull-flag pattern count from 738
    to 298 (-60%) and overall win rate from 21.5% to 22.3% - triangle
    counts were unaffected (already deduplicated in the original run).
  - Overall win rate 22.3% (409 target / 882 stop / 542 time), consistent
    with the 20.0% median win rate among the 129 tickers with at least 5
    patterns - not obviously skewed by a handful of outliers.
  - Triangle (21.6%) and bull flag (25.8%) win rates now show a real gap,
    where the first (buggy) run had called them "nearly identical" -
    a reminder that this kind of infrastructure bug can distort not just
    the totals but the comparisons drawn from them.
  - Individual tickers range from 0% to 67% win rate even among those
    with 5+ patterns, which may just be small-sample noise at that size
    (5-8 examples) rather than a real per-ticker edge - not something to
    read into yet.
  - **Win rate alone hides real information: a pattern's `return_pct` is
    reported alongside it everywhere now** (`main.py --label`'s console
    output, `build_dataset.py`'s `print_summary()`, and the published
    scan artifact), after a real example on WRB showed why - a triangle
    that timed out at +7.2% counted identically to one stopped out at
    -5%, since win rate only credits an exit_reason of "target". Mean
    return doesn't discard that difference: overall mean return across
    all 1,833 patterns is +0.4% (triangle +0.4%, bull flag +0.6%). This
    doesn't change `is_successful`'s definition (still strictly
    `exit_reason == "target"`, matching PLAN.md's original Stage 5
    wording) - mean return is a second, complementary lens, not a
    replacement.
  - **Stage 4's indicator features are now attached to every pattern.**
    `build_labeled_dataset()` runs all five `INDICATOR_COMBINATIONS`
    against each pattern (reusing the exact same functions `main.py
    --indicator N` uses) and joins their output onto that pattern's row,
    with each combination's columns prefixed `indicator{N}_` (e.g.
    combination #1's `is_squeezed` becomes `indicator1_is_squeezed`) so
    all five combinations' columns can sit side by side without name
    collisions.
    - A handful of `indicator5_relative_strength` values come back `NaN`
      for patterns occurring very early in the fetched window - that
      combination needs 63 prior trading days to compute a return
      comparison, which a pattern from a stock's first ~3 months of
      fetched history doesn't have yet. Expected, not a bug; a model
      will need to either drop those rows or impute them.
    - The full-universe run with all five combinations attached produced
      `data/labeled_patterns.csv` at 1,833 rows x 37 columns (the original
      10 labeling/identifying columns plus 27 indicator feature columns
      across the five combinations' `indicator{N}_` prefixes).
  - **A substantive, unexpected finding: more confirmation signals firing
    correlates with *worse* outcomes, not better.** Grouping every pattern
    by `confirmation_count` (how many of the 12 boolean indicator flags
    fired at all, 0-12) shows mean return falling in an almost straight
    line as that count rises - from +2.5% at count=1 down to -1.6% at
    count=9+ (correlation -0.16). Nearly all 12 individual flags show the
    same direction on their own (True performs worse than False), not
    just the combined count. Published as a dedicated artifact, "The
    Confirmation Paradox" (see `SUMMARIES.md`), with the leading
    explanation being that these signals are lagging relative to the
    fixed +10%/-5%/20-day exit rule - by the time several confirm
    together, a meaningful chunk of the move has often already happened,
    leaving less of the fixed target's room ahead and more of the fixed
    stop's room behind. This is read as an argument against a hand-tuned
    rule-based confirmation score (one of PLAN.md's open questions below)
    and for building the actual model next, since a model can weigh - and
    if needed invert - these signals instead of assuming more agreement
    is better.
    Actual model code (logistic regression or gradient boosting, with
    walk-forward validation) is the next step now that the feature
    matrix exists.

### Stage 7 — Backtesting
- Simulate entries on detected + confirmed patterns with realistic slippage
  and transaction costs.
- Evaluate with proper out-of-sample periods, not just in-sample fit.
- **Status:** not started.

---

## Repo Structure (target)

```
Pattern-Detection/
├── .venv/                  # local only, gitignored
├── requirements.txt
├── README.md
├── PLAN.md                 # this file
├── src/
│   ├── patterns.py         # pivot + triangle/flag detection
│   ├── charting.py         # TradingView-style plotting (done)
│   ├── indicators.py       # Stage 4 feature engineering (to build)
│   ├── labeling.py         # Stage 5 outcome labeling (to build)
│   ├── model.py            # Stage 6 training/inference (to build)
│   └── backtest.py         # Stage 7 backtest engine (to build)
├── data/
│   └── ...                 # cached OHLCV, gitignored
├── notebooks/              # exploratory analysis
├── tests/
│   ├── test_patterns.py
│   └── synthetic_data.py   # test fixtures with known patterns injected
└── scripts/
    ├── fetch_real_data.py
    └── demo.py
```

---

## Immediate Next Steps

1. ~~Get the existing `patterns.py` / `fetch_real_data.py` / `charting.py`
   committed and pushed to the repo.~~ Done.
2. Run `fetch_real_data.py` against a handful of real tickers and sanity
   check both the triangle/flag thresholds and the chart output — real
   data will be noisier than the synthetic fixtures this was tuned on.
3. Build `indicators.py`: one function per indicator combo from Stage 4,
   each taking a DataFrame + a detected pattern and returning a feature
   dict. Move the RSI/MACD calculations currently duplicated in
   `charting.py` into this module so there's one source of truth, and have
   `charting.py` import from it.
4. Wire `indicators.py` output into `charting.py`'s `extra_panels` so
   ADX/OBV/ATR/relative-strength show up on the chart alongside RSI/MACD —
   this is what "all the assessment visible in plots" ultimately means once
   Stage 4 exists.
5. Define the labeling rule (Stage 5) precisely before building any model —
   this determines what "success" even means for the project.
6. Set up a lightweight test suite (`pytest`) using the synthetic data
   generator as fixtures, so detector and charting changes can be validated
   quickly without needing live data.

---

## Open Questions / Decisions to Revisit

- Universe: single names, an index constituent list, or ETFs? Affects
  earnings-date handling and relative-strength baseline.
- Exit rule for labeling: fixed target/stop, ATR-based, or trailing?
  Fixed target/stop is what's built (see Stage 5) as the simplest
  starting point; ATR-based and trailing stops remain options to add as
  alternative exit rules once there's a labeled baseline to compare them
  against.
- Position sizing / risk management: out of scope for the detector itself,
  but needed before this becomes a tradeable strategy.
- How much of the ML step is worth it vs. a simpler rule-based scoring
  system (e.g. weighted sum of the 5 indicator combos) — worth prototyping
  the simple version first before investing in a full model.
  - **Partially answered by the Confirmation Paradox finding above (Stage
    6):** a naive weighted-sum score built from these 12 flags would rank
    the *worst*-performing patterns as the most confirmed, since more
    flags firing together correlates with worse outcomes under the
    current exit rule. That doesn't rule out rule-based scoring entirely
    (a score with negative or nonlinear weights could still work), but it
    rules out the simplest version of it, and is itself evidence for
    trying the model rather than assuming a hand-tuned score first.
