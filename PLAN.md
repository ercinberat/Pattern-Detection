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
- **Status:** done (`scripts/fetch_real_data.py: fetch_daily_price_history`) -
  pulls daily OHLCV for a single ticker via yfinance. Still needs a proper
  caching layer and multi-ticker batch support.

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
  `detect_bull_flags`). `detect_triangles()` returns every overlapping
  sliding-window candidate as-is (by design - Stage 4 is where
  confirmation/filtering belongs); a separate `deduplicate_triangles()`
  collapses overlapping candidates down to one per cluster, purely for
  readable charts. Verified visually on real AAPL data via `plot_chart()`'s
  triangle trendline / bull flag pole+box overlays.
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

- **Status:** in progress (`src/indicators.py`). Combination #1
  (Bollinger Band squeeze + volume surge) is built:
  `compute_bollinger_bands()` returns the full band series for charting,
  and `bollinger_squeeze_and_volume_surge(price_data, pattern)` evaluates
  the squeeze/surge at a given pattern's end date and returns a feature
  dict. Wired into `main.py` - prints each detected pattern's features and
  draws the bands on the chart. Combinations #2-5 and the
  swing-trading-specific features are not built yet.

### Stage 5 — Labeling
- For each detected pattern, label the outcome: did price move > X% within
  N bars after the breakout, without first hitting a stop-loss level?
- Needs careful definition to avoid lookahead bias and to reflect a
  realistic swing-trade exit rule (e.g. target/stop/time-based exit).
- **Status:** not started.

### Stage 5b — Visualization
- TradingView-style dark theme, multi-panel chart: candlesticks + volume +
  RSI + MACD stacked below price, shared x-axis.
- Every detected pattern is drawn directly on the price panel: triangle
  trendlines (with r² and contraction ratio labeled), bull flag pole/flag
  zones (with pole return and volume ratio labeled), swing high/low pivot
  markers.
- Built as a single `plot_chart()` entry point taking the DataFrame plus
  detected patterns, so output stays visually consistent everywhere it's
  used.
- `patterns` and `extra_panels` arguments are placeholders wired in for
  Stage 3 and Stage 4 respectively — once triangle/flag detection and
  `indicators.py` exist, their output can be passed straight in without
  changing the charting code.
- Being built incrementally alongside each stage (rather than only at the
  end) so every stage's output can be visually sanity-checked as it's
  built, per an explicit decision to deviate from strict pipeline order.
- **Status:** in progress (`src/charting.py: plot_chart`). Currently renders
  the candlestick + volume chart with Stage 2's pivot markers overlaid.
  Triangle/flag overlays (Stage 3) and RSI/MACD/extra indicator panels
  (Stage 4) not built yet.
- **Display note:** matplotlib's interactive window doesn't reliably render
  scatter overlays (pivot/pattern markers) on this machine — candles/volume
  show fine, but marker artists silently fail to render live even at large
  sizes, while saving to a file renders them correctly every time. So
  `plot_chart()` always renders to a PNG; if no permanent `save_path` is
  given, it saves to a temp file and opens it in the default image viewer
  rather than using matplotlib's interactive window.

### Stage 6 — Modeling
- Start simple: logistic regression or gradient boosting (XGBoost/LightGBM)
  predicting breakout follow-through probability from Stage 4 features.
- Use walk-forward validation (never a random train/test split on time
  series data) to avoid leaking future information.
- **Status:** not started.

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
- Position sizing / risk management: out of scope for the detector itself,
  but needed before this becomes a tradeable strategy.
- How much of the ML step is worth it vs. a simpler rule-based scoring
  system (e.g. weighted sum of the 5 indicator combos) — worth prototyping
  the simple version first before investing in a full model.
