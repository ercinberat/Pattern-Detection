# SUMMARIES.md — Presentation-Style Summaries

A running index of economics/decisions summaries published as Artifacts
for this project, so they're easy to find again from inside the repo
instead of only living in Claude's artifact gallery.

| Summary | Covers | Link |
|---|---|---|
| Signal & Ledger | Stage 4 (five confirmation indicators) and Stage 5 (labeling) — the economic mechanism behind each signal and the real AAPL trade ledger, including the case where the best-confirmed pattern still lost. | https://claude.ai/code/artifact/b56ae841-17db-4ba1-927a-a670238e1d64 |
| S&P 500 Breakout Scan | Stage 6's raw multi-ticker dataset — sortable/searchable per-ticker win rates and mean returns across all 503 S&P 500 constituents (1,833 labeled patterns), plus a win-rate distribution histogram. | https://claude.ai/code/artifact/406c52de-4e05-4953-8d23-b34a6ed53845 |
| The Confirmation Paradox | Stage 6's indicator-impact analysis — across all 1,833 labeled patterns, more of Stage 4's confirmation signals firing correlates with *worse* outcomes, not better; what that means for the fixed exit rule and for building a model instead of a hand-tuned score. | https://claude.ai/code/artifact/f7f2c5fc-b30d-46ff-ad8b-9c6d3217036e |
| Right Signal, Wrong Day | A follow-up to The Confirmation Paradox — measuring the same 12 confirmation flags at each pattern's real breakout day (found via `find_breakout_date()`) instead of its detection window's stale `end_date` flips the confirmation-count/return correlation from -0.145 to +0.106, suggesting the paradox is substantially a measurement-timing artifact. | https://claude.ai/code/artifact/22662028-bb29-4d04-ba0c-d1dce1e1c0a6 |
