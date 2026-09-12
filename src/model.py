"""
Stage 6 modeling: predicts whether a detected pattern's breakout will
follow through (hit its +10% target before its -5% stop, within 20
trading days - see src/labeling.py) using the Stage 4 indicator features
read at the pattern's real breakout day (scripts/build_breakout_dataset.py).

Kept deliberately simple, per PLAN.md's own instruction: logistic
regression was built first. It's a well-understood statistical method
that scores each pattern with a probability of success by fitting a
weight to each feature - "how much does a squeeze, or a high RSI, push
the odds of success up or down" - and its weights can be read directly
afterward, which matters for a project meant to be understood by a
non-software-engineer reader, not just a software one.

Gradient boosting is now also trained, as a second, more flexible model
to compare against that baseline. It builds many small decision trees in
sequence, each one focused on correcting the previous trees' mistakes,
so - unlike logistic regression - it can pick up on interactions between
features (e.g. "signal A only matters when signal B is also true")
without those interactions being written in by hand. This uses
scikit-learn's own GradientBoostingClassifier rather than XGBoost or
LightGBM (the two PLAN.md names as examples) - it's the same underlying
technique, and avoids adding a second, heavier dependency purely to
train a first comparison model with the same, already-small dataset.

Validated with walk-forward splits, never a random train/test split:
patterns are sorted by entry_date and split into chronological folds, so
a fold's model is always trained only on trades that happened *before*
the trades it's being tested on - matching how this would actually be
used (deciding on a pattern with only past data available) and avoiding
the lookahead bias a random split would introduce (training on some of
next month's trades to predict this month's would be cheating).
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Every indicator feature except the six raw price/volume-level columns
# below (indicator3_close, indicator3_donchian_upper, indicator3_obv,
# indicator3_obv_recent_high, indicator4_atr, indicator4_atr_recent_low,
# and indicator2_macd_histogram) - those are absolute dollar prices or
# cumulative share counts, which aren't comparable across tickers at very
# different price levels (a $500 stock's ATR is naturally much bigger
# than a $20 stock's, regardless of which one is the better setup), so a
# model trained across many tickers can't use them meaningfully as-is.
# The boolean flags and the ratio/percentile/index-style numbers
# (band_width_pct, ADX, RSI, relative_strength, etc.) are all already
# scale-independent, so those are kept.
FEATURE_COLUMNS = [
    "indicator1_band_width_pct",
    "indicator1_band_width_percentile",
    "indicator1_is_squeezed",
    "indicator1_volume_ratio",
    "indicator1_is_volume_surge",
    "indicator2_adx",
    "indicator2_plus_di",
    "indicator2_minus_di",
    "indicator2_is_trending",
    "indicator2_is_bullish_direction",
    "indicator2_is_macd_bullish_flip",
    "indicator3_is_donchian_breakout",
    "indicator3_is_obv_new_high",
    "indicator4_rsi",
    "indicator4_was_basing",
    "indicator4_is_momentum_shift",
    "indicator4_is_atr_expanding",
    "indicator5_pct_from_52_week_high",
    "indicator5_is_near_52_week_high",
    "indicator5_relative_strength",
    "indicator5_is_outperforming_benchmark",
]

TARGET_COLUMN = "is_successful"


def load_training_data(csv_path: str = "data/breakout_labeled_patterns.csv"):
    """
    Load scripts/build_breakout_dataset.py's output and prepare it for
    training: pick out FEATURE_COLUMNS plus a pattern_type feature, drop
    any pattern missing a feature value, and sort everything by
    entry_date so later code can split it into chronological folds.

    A pattern is missing indicator5_relative_strength (needs 63 prior
    trading days of history) far more often than any other feature - see
    PLAN.md's Stage 6 notes - so dropping any row with a missing feature,
    rather than trying to fill it in, is the simplest first-pass choice;
    imputing it is a reasonable thing to try later.

    Returns (features, target, entry_dates) - three same-length,
    same-order pandas objects: a DataFrame of feature columns, a Series
    of True/False labels, and a Series of entry dates for splitting.
    """
    dataset = pd.read_csv(csv_path)
    dataset["entry_date"] = pd.to_datetime(dataset["entry_date"], utc=True)

    # A pattern's type is known before its outcome, so it's a legitimate
    # feature - encoded as a single True/False column since there are
    # only two pattern types.
    dataset["is_bull_flag_pattern"] = dataset["pattern_type"] == "bull_flag"
    feature_columns_with_pattern_type = FEATURE_COLUMNS + ["is_bull_flag_pattern"]

    dataset = dataset.dropna(subset=feature_columns_with_pattern_type)
    dataset = dataset.sort_values("entry_date").reset_index(drop=True)

    features = dataset[feature_columns_with_pattern_type].astype(float)
    target = dataset[TARGET_COLUMN].astype(bool)
    entry_dates = dataset["entry_date"]

    return features, target, entry_dates


def build_logistic_regression_model() -> Pipeline:
    """
    A logistic regression, preceded by StandardScaler (rescales every
    feature to the same rough range - mean 0, spread 1 - before fitting).
    Without that rescaling, a feature like ADX (0-100) would dominate the
    fit purely because its numbers are bigger than, say,
    relative_strength's (usually close to 1.0), not because it actually
    matters more. Tree-based models like build_gradient_boosting_model()
    below don't need this - see its own docstring for why.
    """
    return Pipeline([("scale", StandardScaler()), ("logistic_regression", LogisticRegression())])


def build_gradient_boosting_model() -> GradientBoostingClassifier:
    """
    A gradient boosting classifier - see this module's own docstring for
    what that means and why it's being tried as a second model.
    random_state is fixed so a run's results don't change from one run to
    the next just by chance. Doesn't need StandardScaler first: a
    decision tree splits each feature at a threshold ("is ADX above
    27.4?"), and where exactly that threshold lands doesn't depend on
    what scale the feature's numbers happen to be in.
    """
    return GradientBoostingClassifier(random_state=0)


def train_and_evaluate(features: pd.DataFrame, target: pd.Series, n_splits: int = 5, build_model=build_logistic_regression_model) -> pd.DataFrame:
    """
    Train and test a model across n_splits walk-forward folds, and return
    one row of results per fold.

    build_model: a no-argument function returning a fresh, untrained
        scikit-learn model each time it's called - build_logistic_regression_model
        (the default) or build_gradient_boosting_model above. A fresh
        model is built for every fold rather than reusing one instance,
        so nothing from an earlier fold's fit can leak into the next.

    features/target must already be sorted by entry_date (see
    load_training_data()) - TimeSeriesSplit cuts the data into n_splits+1
    chronological chunks and, for fold i, trains on every chunk up
    through i and tests on chunk i+1. Each fold's test set is entirely
    later in time than everything its model was trained on.

    For each fold, reports:
        train_size/test_size: how many patterns were in each.
        baseline_win_rate: the test fold's actual win rate - what you'd
            expect to earn by taking every pattern with no filtering at
            all, the number the model needs to beat to be useful.
        model_win_rate_if_predicted_positive: the actual win rate among
            only the patterns the model predicted would succeed (using a
            plain 50% probability cutoff) - compare this to
            baseline_win_rate to see whether trusting the model's yes/no
            call would have actually picked better trades.
        top_20pct_win_rate: the actual win rate among just the 20% of
            patterns the model was most confident about - a stricter,
            more realistic use of the model than "trade everything it
            says yes to," since a probability score is naturally more
            useful for ranking candidates than for a hard yes/no cutoff.
        accuracy/precision/recall/roc_auc: standard classification
            metrics, for readers who want the conventional numbers too.
            roc_auc is the most informative single number here - 0.5
            means the model is no better than a coin flip, 1.0 means it
            perfectly separates winners from losers.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_number, (train_positions, test_positions) in enumerate(splitter.split(features), start=1):
        train_features, test_features = features.iloc[train_positions], features.iloc[test_positions]
        train_target, test_target = target.iloc[train_positions], target.iloc[test_positions]

        model = build_model()
        model.fit(train_features, train_target)
        predicted_probabilities = model.predict_proba(test_features)[:, 1]
        predicted_labels = predicted_probabilities >= 0.5

        # The top 20% most-confident predictions, regardless of whether
        # they cross the 0.5 cutoff - ranking candidates is usually a
        # more realistic use of a probability score than a hard yes/no.
        top_20pct_cutoff = pd.Series(predicted_probabilities).quantile(0.8)
        top_20pct_mask = predicted_probabilities >= top_20pct_cutoff

        fold_results.append(
            {
                "fold": fold_number,
                "train_size": len(train_features),
                "test_size": len(test_features),
                "baseline_win_rate": test_target.mean(),
                "model_win_rate_if_predicted_positive": (
                    test_target[predicted_labels].mean() if predicted_labels.any() else float("nan")
                ),
                "top_20pct_win_rate": test_target[top_20pct_mask].mean(),
                "accuracy": accuracy_score(test_target, predicted_labels),
                "precision": precision_score(test_target, predicted_labels, zero_division=0),
                "recall": recall_score(test_target, predicted_labels, zero_division=0),
                "roc_auc": roc_auc_score(test_target, predicted_probabilities),
            }
        )

    return pd.DataFrame(fold_results)


def print_feature_weights(features: pd.DataFrame, target: pd.Series) -> None:
    """
    Fit one logistic regression on the FULL dataset (not a single fold)
    and print each feature's learned weight, so the model's reasoning can
    be read directly rather than treated as a black box - the whole point
    of starting with logistic regression instead of gradient boosting.

    A positive weight means that feature pushes the predicted probability
    of success up as it increases (for a True/False feature: up when the
    feature is True); a negative weight pushes it down. Weights are
    comparable to each other here because features were standardized
    first (see build_logistic_regression_model()'s docstring) - without
    that, a bigger weight could just mean "this feature's raw numbers are
    smaller," not "this feature matters more."
    """
    model_pipeline = build_logistic_regression_model()
    model_pipeline.fit(features, target)

    weights = pd.Series(
        model_pipeline.named_steps["logistic_regression"].coef_[0],
        index=features.columns,
    ).sort_values()

    print("Logistic regression feature weights (fit on all data, standardized - positive pushes toward 'successful'):")
    print(weights.to_string(float_format="{:+.3f}".format))


def print_feature_importances(features: pd.DataFrame, target: pd.Series) -> None:
    """
    Fit one gradient boosting model on the FULL dataset (not a single
    fold) and print each feature's importance - how much that feature
    reduced prediction error, summed across every tree split that used
    it. Unlike logistic regression's weights, importances don't have a
    direction (there's no equivalent of "positive" or "negative" here,
    since a tree can use a feature differently at different splits) and
    they're always zero or positive, and add up to 1.0 across all
    features - a feature with importance 0.15 accounted for 15% of the
    model's total error reduction.
    """
    gradient_boosting_model = build_gradient_boosting_model()
    gradient_boosting_model.fit(features, target)

    importances = pd.Series(
        gradient_boosting_model.feature_importances_,
        index=features.columns,
    ).sort_values(ascending=False)

    print("Gradient boosting feature importances (fit on all data, sum to 1.0):")
    print(importances.to_string(float_format="{:.3f}".format))


def _print_fold_results(model_name: str, fold_results: pd.DataFrame) -> None:
    """Print one model's per-fold results table plus its across-fold averages, in a consistent format."""
    print(f"--- {model_name}: walk-forward validation ({len(fold_results)} chronological folds) ---")
    print(
        fold_results.to_string(
            index=False,
            formatters={
                "baseline_win_rate": "{:.1%}".format,
                "model_win_rate_if_predicted_positive": "{:.1%}".format,
                "top_20pct_win_rate": "{:.1%}".format,
                "accuracy": "{:.1%}".format,
                "precision": "{:.1%}".format,
                "recall": "{:.1%}".format,
                "roc_auc": "{:.3f}".format,
            },
        )
    )
    print(
        f"Averaged across folds: baseline win rate {fold_results['baseline_win_rate'].mean():.1%}, "
        f"model win rate (predicted positive) {fold_results['model_win_rate_if_predicted_positive'].mean():.1%}, "
        f"top-20%-confidence win rate {fold_results['top_20pct_win_rate'].mean():.1%}, "
        f"mean ROC-AUC {fold_results['roc_auc'].mean():.3f}\n"
    )


if __name__ == "__main__":
    features, target, entry_dates = load_training_data()
    print(f"Loaded {len(features)} patterns with complete features, {entry_dates.min().date()} to {entry_dates.max().date()}")
    print(f"Overall win rate: {target.mean():.1%}\n")

    logistic_regression_results = train_and_evaluate(features, target, build_model=build_logistic_regression_model)
    _print_fold_results("Logistic regression", logistic_regression_results)

    gradient_boosting_results = train_and_evaluate(features, target, build_model=build_gradient_boosting_model)
    _print_fold_results("Gradient boosting", gradient_boosting_results)

    print(
        f"Mean ROC-AUC comparison: logistic regression {logistic_regression_results['roc_auc'].mean():.3f} "
        f"vs. gradient boosting {gradient_boosting_results['roc_auc'].mean():.3f}\n"
    )

    print_feature_weights(features, target)
    print()
    print_feature_importances(features, target)
