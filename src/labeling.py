"""
Stage 5 labeling: deciding, for each detected pattern, whether its
breakout actually worked - the ground truth that Stage 6's model will
eventually be trained to predict, and the same ground truth Stage 3's
"Next" note says is needed before its detection thresholds can be
properly validated (see PLAN.md).

The exit rule used here is a fixed target/stop, chosen in PLAN.md's
Stage 5 "Open Questions" as the simplest starting point - hit a target
% gain within a maximum holding period, without first hitting a stop %
loss. ATR-based and trailing-stop exit rules were discussed as
alternatives and deliberately left out of this first version; they're
still listed as options to add later.
"""

import pandas as pd

from src.patterns import BullFlagPattern, TrianglePattern, pattern_evaluation_date


def label_pattern_outcome(
    price_data: pd.DataFrame,
    pattern,
    target_pct: float = 10.0,
    stop_pct: float = 5.0,
    max_holding_days: int = 20,
) -> dict:
    """
    Label whether a detected pattern's breakout followed through with a
    real move or failed, using a fixed target/stop/time-based exit rule.

    The entry is assumed to happen at the Open of the bar right AFTER the
    pattern's end date - not on the pattern's own end-date bar, since
    that bar's own Close is part of what the pattern detection is based
    on, so it isn't realistic to assume a trade could be entered before
    that bar has even finished. This also means the label only ever looks
    at price data strictly after the pattern's end date, never at data
    the detection itself already used - avoiding the lookahead bias
    PLAN.md's Stage 5 description calls out.

    target_pct: how far price has to rise (as a % of the entry price) to
        count as the target being hit - a successful breakout.
    stop_pct: how far price can fall (as a % of the entry price) before
        the trade is considered stopped out - a failed breakout.
    max_holding_days: the maximum number of bars to hold the trade before
        giving up and exiting at whatever price is then (a time-based
        exit), if neither the target nor the stop was hit first.

    Returns a dict:
        entry_date, entry_price: when and at what price the trade starts.
        exit_date, exit_price: when and at what price the trade ends.
        exit_reason: "target", "stop", or "time" - whichever triggered
            the exit.
        return_pct: the % return from entry_price to exit_price.
        is_successful: True only if exit_reason is "target" - matching
            PLAN.md's definition of a successful label ("price move > X%
            ... without first hitting a stop-loss"). A "time" exit that
            happens to be up a little, but never reached the target, is
            not counted as a success under this definition.

    Returns None if there isn't at least one bar of price data after the
    pattern's end date to label against yet (e.g. the pattern's end date
    is the most recent bar in price_data).
    """
    evaluation_date = pattern_evaluation_date(pattern)
    end_date_position = price_data.index.get_loc(evaluation_date)

    entry_position = end_date_position + 1
    if entry_position >= len(price_data):
        return None

    entry_date = price_data.index[entry_position]
    entry_price = price_data["Open"].iloc[entry_position]
    target_price = entry_price * (1 + target_pct / 100)
    stop_price = entry_price * (1 - stop_pct / 100)

    last_position = min(entry_position + max_holding_days, len(price_data) - 1)

    for position in range(entry_position, last_position + 1):
        bar = price_data.iloc[position]

        # A single daily bar's range can cover both the stop and target
        # prices, but daily bars don't say which was actually touched
        # first within the day. Checking the stop first means the worse
        # outcome is assumed whenever it's ambiguous, which is the safer
        # assumption for labeling (it doesn't overstate how often
        # breakouts succeed).
        if bar["Low"] <= stop_price:
            return _build_label(entry_date, entry_price, price_data.index[position], stop_price, "stop")
        if bar["High"] >= target_price:
            return _build_label(entry_date, entry_price, price_data.index[position], target_price, "target")

    exit_price = price_data["Close"].iloc[last_position]
    return _build_label(entry_date, entry_price, price_data.index[last_position], exit_price, "time")


def _build_label(entry_date, entry_price, exit_date, exit_price, exit_reason: str) -> dict:
    """Assemble one label_pattern_outcome() result dict, shared by all three exit branches."""
    return {
        "entry_date": entry_date,
        "entry_price": entry_price,
        "exit_date": exit_date,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "return_pct": (exit_price - entry_price) / entry_price * 100,
        "is_successful": exit_reason == "target",
    }


def label_patterns(price_data: pd.DataFrame, patterns: list, **label_kwargs) -> list[dict]:
    """
    Label the outcome of every pattern in `patterns`, skipping any
    pattern too close to the end of price_data to have a full bar of
    history after it yet.

    **label_kwargs are passed straight through to label_pattern_outcome()
    (target_pct, stop_pct, max_holding_days).

    Returns a list of dicts, each label_pattern_outcome() result plus:
        pattern: the TrianglePattern/BullFlagPattern the label came from.
        pattern_type: "triangle" or "bull_flag".
    so a label can always be traced back to what produced it - e.g. for
    the threshold-validation sweep described in PLAN.md's Stage 3 "Next"
    note.
    """
    labels = []
    for pattern in patterns:
        outcome = label_pattern_outcome(price_data, pattern, **label_kwargs)
        if outcome is None:
            continue
        pattern_type = "triangle" if isinstance(pattern, TrianglePattern) else "bull_flag"
        labels.append({"pattern": pattern, "pattern_type": pattern_type, **outcome})
    return labels


if __name__ == "__main__":
    # Quick manual check: fetch AAPL's daily history, detect its
    # patterns, and print the label for each one, so the outcomes can be
    # sanity-checked against the chart by eye.
    from scripts.fetch_real_data import fetch_daily_price_history
    from src.patterns import deduplicate_bull_flags, deduplicate_triangles, detect_bull_flags, detect_triangles, find_pivots

    aapl_daily_data = fetch_daily_price_history("AAPL")
    aapl_pivots = find_pivots(aapl_daily_data, order=5)
    aapl_triangles = deduplicate_triangles(detect_triangles(aapl_pivots))
    aapl_bull_flags = deduplicate_bull_flags(detect_bull_flags(aapl_daily_data))

    aapl_labels = label_patterns(aapl_daily_data, aapl_triangles + aapl_bull_flags)
    for label in aapl_labels:
        pattern = label["pattern"]
        pattern_description = (
            f"{pattern.triangle_type} triangle ending {pattern.end_date.date()}"
            if isinstance(pattern, TrianglePattern)
            else f"bull flag ending {pattern.flag_end_date.date()}"
        )
        print(f"{pattern_description}: {label['exit_reason']} exit, return {label['return_pct']:.1f}%, successful={label['is_successful']}")
