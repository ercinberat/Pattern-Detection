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
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# How many inner, earlier-than-the-outer-fold splits a nested
# hyperparameter search gets to tune on, inside each of
# train_and_evaluate()'s own outer walk-forward folds - see
# build_logistic_regression_model()/build_gradient_boosting_model()'s
# docstrings for why the search has to be nested, not run once up front.
_INNER_TUNING_SPLITS = 3

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
# Stage 3's own pattern-quality measures (how well-fit a triangle's
# trendlines are, how sharp a bull flag's pole was) and two simple
# market-regime descriptors (see scripts/build_breakout_dataset.py's
# _compute_market_regime_series()) - both were sitting unused until now.
# A barely-qualifying pattern and a tight, clean one used to look
# identical to the model; so did a breakout in a trending market and one
# in a choppy, directionless one.
GEOMETRIC_AND_MARKET_COLUMNS = [
    "pattern_high_r_squared",
    "pattern_low_r_squared",
    "pattern_contraction_pct",
    "pattern_pole_return_pct",
    "pattern_flag_volume_ratio",
    "pattern_retracement_pct",
    "market_pct_from_50d_average",
    "market_20d_volatility_pct",
]

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
] + GEOMETRIC_AND_MARKET_COLUMNS

# The continuous/ratio-style features out of FEATURE_COLUMNS above -
# everything except the 12 True/False confirmation flags. Split out as
# its own list to directly test a hypothesis gradient boosting's feature
# importances raised (see PLAN.md's Stage 6 notes): those 12 booleans
# accounted for only 2.8% of its total importance combined, so maybe the
# raw indicator readings carry the real signal and the hand-picked
# True/False thresholds are just discarding information, not adding any.
CONTINUOUS_FEATURE_COLUMNS = [
    "indicator1_band_width_pct",
    "indicator1_band_width_percentile",
    "indicator1_volume_ratio",
    "indicator2_adx",
    "indicator2_plus_di",
    "indicator2_minus_di",
    "indicator4_rsi",
    "indicator5_pct_from_52_week_high",
    "indicator5_relative_strength",
] + GEOMETRIC_AND_MARKET_COLUMNS

TARGET_COLUMN = "is_successful"

# Columns imputed with a fixed, neutral fill value instead of being
# dropped, wherever they're missing - both are only missing because a
# pattern's breakout landed too early in its own fetched price history
# for that column's lookback to be complete yet (63 trading days for
# relative_strength, 50 for the market-regime average), not because
# anything went wrong. The fill value is the number that means "neutral"
# for that column (performing exactly in line with the benchmark; sitting
# exactly at the market's own average), not a guess at the true value.
IMPUTED_FEATURE_FILL_VALUES = {
    "indicator5_relative_strength": 1.0,
    "market_pct_from_50d_average": 0.0,
}


def load_training_data(csv_path: str = "data/breakout_labeled_patterns.csv", feature_columns: list = None):
    """
    Load scripts/build_breakout_dataset.py's output and prepare it for
    training: pick out feature_columns plus a pattern_type feature, drop
    any pattern missing a feature value, and sort everything by
    entry_date so later code can split it into chronological folds.

    feature_columns: which indicator columns to use - FEATURE_COLUMNS
        (the default, if not given) for every feature, or
        CONTINUOUS_FEATURE_COLUMNS to test the 12 True/False confirmation
        flags' own contribution by leaving them out entirely.

    Two columns (see IMPUTED_FEATURE_FILL_VALUES) are filled with a
    neutral value instead of being dropped, since they're only ever
    missing because a pattern's breakout landed too early in its own
    fetched history for that column's lookback window to be complete -
    not because anything is actually wrong with that row. Any other
    missing feature value still causes the whole row to be dropped, as
    the simplest first-pass choice.

    Returns (features, target, entry_dates, return_pct) - four
    same-length, same-order pandas objects: a DataFrame of feature
    columns, a Series of True/False "hit target" labels, a Series of
    entry dates for splitting, and a Series of the pattern's actual %
    return - the magnitude-aware alternative to `target`, for
    train_and_evaluate_regression() below.
    """
    if feature_columns is None:
        feature_columns = FEATURE_COLUMNS

    dataset = pd.read_csv(csv_path)
    dataset["entry_date"] = pd.to_datetime(dataset["entry_date"], utc=True)

    # A pattern's type is known before its outcome, so it's a legitimate
    # feature - encoded as a single True/False column since there are
    # only two pattern types. Kept in both the full and continuous-only
    # feature sets, since it isn't one of the 12 confirmation flags being
    # tested - it's which pattern shape was detected, not a signal about
    # whether the breakout will hold.
    dataset["is_bull_flag_pattern"] = dataset["pattern_type"] == "bull_flag"
    feature_columns_with_pattern_type = feature_columns + ["is_bull_flag_pattern"]

    for column, fill_value in IMPUTED_FEATURE_FILL_VALUES.items():
        if column in feature_columns_with_pattern_type:
            dataset[column] = dataset[column].fillna(fill_value)

    dataset = dataset.dropna(subset=feature_columns_with_pattern_type)
    dataset = dataset.sort_values("entry_date").reset_index(drop=True)

    features = dataset[feature_columns_with_pattern_type].astype(float)
    target = dataset[TARGET_COLUMN].astype(bool)
    entry_dates = dataset["entry_date"]
    return_pct = dataset["return_pct"].astype(float)

    return features, target, entry_dates, return_pct


def build_logistic_regression_model() -> GridSearchCV:
    """
    A logistic regression, preceded by StandardScaler (rescales every
    feature to the same rough range - mean 0, spread 1 - before fitting).
    Without that rescaling, a feature like ADX (0-100) would dominate the
    fit purely because its numbers are bigger than, say,
    relative_strength's (usually close to 1.0), not because it actually
    matters more. Tree-based models like build_gradient_boosting_model()
    below don't need this - see its own docstring for why.

    class_weight="balanced" reweights the loss for the dataset's own
    class imbalance (roughly 25% successful, 75% not) - without it, a
    model can score reasonably well by leaning toward "not successful"
    most of the time, without actually learning much about what tells the
    two apart.

    Wrapped in GridSearchCV to search over C (how strongly the model is
    penalized for large weights - smaller C means a simpler, more
    conservative fit) using TimeSeriesSplit as the *inner* cross-
    validation. This has to be nested inside each of
    train_and_evaluate()'s own outer walk-forward folds (calling
    build_logistic_regression_model() fresh per fold, then fitting on
    just that fold's training rows) rather than tuned once up front on
    all the data - tuning on the whole dataset first would let the
    search see rows from after some folds' test periods, which is the
    same lookahead bias walk-forward validation exists to avoid in the
    model itself.
    """
    pipeline = Pipeline([("scale", StandardScaler()), ("logistic_regression", LogisticRegression(class_weight="balanced"))])
    parameter_grid = {"logistic_regression__C": [0.01, 0.1, 1.0, 10.0]}
    return GridSearchCV(pipeline, param_grid=parameter_grid, cv=TimeSeriesSplit(n_splits=_INNER_TUNING_SPLITS), scoring="roc_auc")


def build_gradient_boosting_model() -> GridSearchCV:
    """
    A gradient boosting classifier - see this module's own docstring for
    what that means and why it's being tried as a second model.
    random_state is fixed so a run's results don't change from one run to
    the next just by chance. Doesn't need StandardScaler first: a
    decision tree splits each feature at a threshold ("is ADX above
    27.4?"), and where exactly that threshold lands doesn't depend on
    what scale the feature's numbers happen to be in.

    Wrapped in GridSearchCV, same as build_logistic_regression_model()
    above and for the same nested-walk-forward reason, searching over
    tree depth, learning rate, number of trees, and the minimum patterns
    required in a leaf before it's allowed to split further (a small
    guard against a tree fitting to just a handful of rows). Left at
    scikit-learn's defaults everywhere, an untuned gradient boosting
    model isn't a fair comparison against a logistic regression that's
    at least had its own regularization strength tuned.
    """
    parameter_grid = {
        "max_depth": [2, 3],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [50, 100],
        "min_samples_leaf": [10, 30],
    }
    return GridSearchCV(
        GradientBoostingClassifier(random_state=0),
        param_grid=parameter_grid,
        cv=TimeSeriesSplit(n_splits=_INNER_TUNING_SPLITS),
        scoring="roc_auc",
    )


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


def build_linear_regression_model() -> Pipeline:
    """
    A plain linear regression predicting return_pct directly (a number)
    instead of is_successful (True/False) - the magnitude-aware
    counterpart to build_logistic_regression_model() (see this module's
    docstring, and PLAN.md's Stage 6 notes on why the binary label throws
    away outcome magnitude - a pattern that timed out at +9% and one
    stopped out at -5% are worlds apart, but count as the same "not
    successful" label). StandardScaler first, for the same reason as the
    classifier version.
    """
    return Pipeline([("scale", StandardScaler()), ("linear_regression", LinearRegression())])


def build_gradient_boosting_regressor() -> GradientBoostingRegressor:
    """
    The magnitude-aware counterpart to build_gradient_boosting_model().
    Left at scikit-learn's defaults, not wrapped in a hyperparameter
    search like the classifiers above - this regression version is a
    first exploratory pass at the magnitude-aware idea (see PLAN.md's
    Stage 6 notes), not yet a model being seriously tuned and compared.
    """
    return GradientBoostingRegressor(random_state=0)


def train_and_evaluate_regression(
    features: pd.DataFrame, return_pct: pd.Series, n_splits: int = 5, build_model=build_linear_regression_model
) -> pd.DataFrame:
    """
    Same walk-forward idea as train_and_evaluate() above, but predicting
    return_pct directly instead of is_successful.

    Reports, per fold:
        baseline_mean_return: the test fold's actual mean return_pct -
            what you'd earn taking every pattern with no filtering at
            all, the number this needs to beat to be useful.
        top_20pct_predicted_mean_return: the actual mean return_pct among
            just the 20% of patterns the model predicted would return the
            most - the realistic way to use a return prediction is to
            rank candidates by it, not to trust any single predicted
            number exactly.
        correlation: how well the model's predicted return_pct tracks the
            actual return_pct on the test fold (Pearson correlation, -1
            to +1, 0 meaning no relationship) - independent of scale, so
            it's comparable fold to fold even though actual returns vary.
    """
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_results = []

    for fold_number, (train_positions, test_positions) in enumerate(splitter.split(features), start=1):
        train_features, test_features = features.iloc[train_positions], features.iloc[test_positions]
        train_return, test_return = return_pct.iloc[train_positions], return_pct.iloc[test_positions]

        model = build_model()
        model.fit(train_features, train_return)
        predicted_return = pd.Series(model.predict(test_features), index=test_return.index)

        top_20pct_cutoff = predicted_return.quantile(0.8)
        top_20pct_mask = predicted_return >= top_20pct_cutoff

        fold_results.append(
            {
                "fold": fold_number,
                "train_size": len(train_features),
                "test_size": len(test_features),
                "baseline_mean_return": test_return.mean(),
                "top_20pct_predicted_mean_return": test_return[top_20pct_mask].mean(),
                "correlation": predicted_return.corr(test_return),
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
    # build_logistic_regression_model() returns a GridSearchCV wrapper
    # (see its own docstring for why); .fit() runs its internal tuning
    # and refits the best-found settings on all of `features`/`target`,
    # available afterward as .best_estimator_.
    search = build_logistic_regression_model()
    search.fit(features, target)
    best_pipeline = search.best_estimator_

    weights = pd.Series(
        best_pipeline.named_steps["logistic_regression"].coef_[0],
        index=features.columns,
    ).sort_values()

    print(f"Logistic regression feature weights (best C={best_pipeline.named_steps['logistic_regression'].C}, fit on all data, standardized - positive pushes toward 'successful'):")
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
    # See print_feature_weights()'s comment above - same GridSearchCV
    # wrapper, same .best_estimator_ unwrapping.
    search = build_gradient_boosting_model()
    search.fit(features, target)
    best_model = search.best_estimator_

    importances = pd.Series(
        best_model.feature_importances_,
        index=features.columns,
    ).sort_values(ascending=False)

    print(f"Gradient boosting feature importances (best params={search.best_params_}, fit on all data, sum to 1.0):")
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


def _run_comparison(feature_set_name: str, feature_columns: list) -> dict:
    """
    Load the data with the given feature_columns, train+evaluate both
    models, print their per-fold tables, and return each model's mean
    ROC-AUC so _run_comparison()'s caller can put every feature
    set/model combination side by side in one final summary table.
    """
    features, target, entry_dates, _ = load_training_data(feature_columns=feature_columns)
    print(
        f"=== Feature set: {feature_set_name} ({len(feature_columns) + 1} features) === "
        f"{len(features)} patterns, {entry_dates.min().date()} to {entry_dates.max().date()}, "
        f"overall win rate {target.mean():.1%}\n"
    )

    logistic_regression_results = train_and_evaluate(features, target, build_model=build_logistic_regression_model)
    _print_fold_results("Logistic regression", logistic_regression_results)

    gradient_boosting_results = train_and_evaluate(features, target, build_model=build_gradient_boosting_model)
    _print_fold_results("Gradient boosting", gradient_boosting_results)

    return {
        "feature_set": feature_set_name,
        "logistic_regression_roc_auc": logistic_regression_results["roc_auc"].mean(),
        "gradient_boosting_roc_auc": gradient_boosting_results["roc_auc"].mean(),
    }


if __name__ == "__main__":
    # Run the same two models on two feature sets - every feature, and
    # just the 9 continuous/ratio ones with all 12 confirmation booleans
    # left out - to test the hypothesis gradient boosting's feature
    # importances raised: that those booleans might be pure noise riding
    # along with the continuous readings, not adding real signal.
    summary_rows = [
        _run_comparison("All features", FEATURE_COLUMNS),
        _run_comparison("Continuous only (no confirmation flags)", CONTINUOUS_FEATURE_COLUMNS),
    ]

    print("=== Summary: mean ROC-AUC by feature set ===")
    print(pd.DataFrame(summary_rows).to_string(index=False, float_format="{:.3f}".format))

    print()
    features, target, _, return_pct = load_training_data()
    print_feature_weights(features, target)
    print()
    print_feature_importances(features, target)

    print("\n=== Magnitude-aware target: predicting return_pct directly instead of is_successful ===\n")
    linear_regression_results = train_and_evaluate_regression(features, return_pct, build_model=build_linear_regression_model)
    print("--- Linear regression ---")
    print(linear_regression_results.to_string(index=False, float_format="{:.2f}".format))
    print(
        f"Averaged: baseline mean return {linear_regression_results['baseline_mean_return'].mean():+.2f}%, "
        f"top-20%-predicted mean return {linear_regression_results['top_20pct_predicted_mean_return'].mean():+.2f}%, "
        f"mean correlation {linear_regression_results['correlation'].mean():+.3f}\n"
    )

    gradient_boosting_regression_results = train_and_evaluate_regression(
        features, return_pct, build_model=build_gradient_boosting_regressor
    )
    print("--- Gradient boosting regressor ---")
    print(gradient_boosting_regression_results.to_string(index=False, float_format="{:.2f}".format))
    print(
        f"Averaged: baseline mean return {gradient_boosting_regression_results['baseline_mean_return'].mean():+.2f}%, "
        f"top-20%-predicted mean return {gradient_boosting_regression_results['top_20pct_predicted_mean_return'].mean():+.2f}%, "
        f"mean correlation {gradient_boosting_regression_results['correlation'].mean():+.3f}"
    )
