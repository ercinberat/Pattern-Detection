# CLAUDE.md

This file is read automatically at the start of every Claude Code session
in this repo. Read `PLAN.md` and `STYLE.md` in full before making any
changes — they are not optional background reading, they are the spec and
the rulebook for this project.

---

## What this project is

A tool that scans daily-bar stock price data for momentum breakout setups —
triangles and bullish flags — and assesses whether a breakout is likely to
follow through, for **swing trading** (holding periods of days to a few
weeks). See `PLAN.md` for the full pipeline (data → pivots → pattern
detection → indicator confirmation → labeling → modeling → backtesting →
visualization) and current status of each stage.

## Before you write any code

1. Read `PLAN.md`. Know which stage you're working on and what it says to
   build. Don't build ahead of the plan or substitute a different approach
   — if you think the plan should change, ask first, don't just do it.
2. Read `STYLE.md`. It governs how code gets written in this repo:
   - Never delete anything without asking first.
   - Stick to the plan; ask before improving on it.
   - Write for a technically capable engineer who is not a software
     developer — clear, readable, minimal abstraction.
   - Use descriptive names, not abbreviations.
   - Comment functions and non-obvious logic clearly and often.
3. If either file is missing, out of date, or contradicts what you're being
   asked to do, say so before proceeding — don't silently pick one over the
   other.

## Repo layout

```
Pattern-Detection/
├── .venv/                  # local only, gitignored
├── requirements.txt
├── README.md
├── PLAN.md                 # what to build, in what order, current status
├── STYLE.md                # how to write code here
├── CLAUDE.md                # this file
├── src/
│   ├── patterns.py         # pivot + triangle/flag detection
│   ├── charting.py         # TradingView-style plotting
│   ├── indicators.py       # confirmation indicators (Stage 4)
│   ├── labeling.py         # outcome labeling (Stage 5)
│   ├── model.py             # training/inference (Stage 6)
│   └── backtest.py         # backtest engine (Stage 7)
├── data/                   # cached OHLCV, gitignored
├── notebooks/
├── tests/
│   ├── test_patterns.py
│   └── synthetic_data.py   # test fixtures with known patterns injected
└── scripts/
    ├── fetch_real_data.py
    └── demo.py
```

## Working agreement

- Ask before deleting files, functions, or working code.
- Ask before pushing to `main`. Prefer a branch + review for anything
  beyond a small fix.
- Ask before changing scope, approach, or design versus what `PLAN.md`
  describes for the current stage.
- After finishing a working piece of a stage, update `PLAN.md`'s status
  for that stage before moving on, so the plan stays an accurate record of
  where the project actually is.
