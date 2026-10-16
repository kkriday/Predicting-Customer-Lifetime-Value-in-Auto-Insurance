"""End-to-end CLV pipeline: structure analysis, tuned models, error by tier,
conformal prediction intervals, permutation importance.

Usage:
    python run_pipeline.py

Outputs land in reports/ (metrics.json, model_comparison.csv, figures/).
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import (KFold, RandomizedSearchCV, cross_val_predict,
                                     cross_validate, train_test_split)

from src import plots
from src.config import (CV_FOLDS, INTERVAL_COVERAGE, REPORTS_DIR, ROOT,
                        SEARCH_ITERATIONS, SEED, TARGET, TEST_SIZE)
from src.data import load, policy_tier, split_xy
from src.evaluate import (TierConformal, error_by_tier, interval_report,
                          regression_metrics)
from src.models import candidates

warnings.filterwarnings("ignore", category=UserWarning)
BASELINE = "Baseline (mean CLV)"
SIMPLICITY_ORDER = ["Ridge Regression", "Random Forest", "LightGBM", "XGBoost"]
SCORING = {
    "mae": "neg_mean_absolute_error",
    "r2_dollar": "r2",
    "r2_log": make_scorer(lambda y, p: r2_score(np.log(y), np.log(np.clip(p, 1, None)))),
}


def two_policy_band_r2(df):
    """How much of the 2-policy CLV multiplier can the other attributes explain?"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score

    two = df[df["Number of Policies"] == 2]
    ratio = two[TARGET] / two["Monthly Premium Auto"]
    X = pd.get_dummies(two.drop(columns=["Customer", TARGET, "Effective To Date"]), drop_first=True)
    rf = RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1)
    return float(cross_val_score(rf, X, ratio, cv=KFold(CV_FOLDS, shuffle=True, random_state=SEED),
                                 scoring="r2").mean())


def structure(df):
    tier = policy_tier(df["Number of Policies"])
    plots.clv_structure(df, tier)
    ratio = df[TARGET] / df["Monthly Premium Auto"]
    by_tier = ratio.groupby(tier).describe(percentiles=[0.05, 0.5, 0.95]).round(1)
    return {
        "rows": int(len(df)),
        "clv_mean": round(float(df[TARGET].mean()), 0),
        "clv_median": round(float(df[TARGET].median()), 0),
        "clv_skew": round(float(df[TARGET].skew()), 2),
        "tier_share": tier.value_counts(normalize=True).round(4).to_dict(),
        "tier_mean_clv": df.groupby(tier)[TARGET].mean().round(0).to_dict(),
        "clv_per_premium_dollar": by_tier[["5%", "50%", "95%", "std"]].to_dict("index"),
        "zero_income_share": round(float((df["Income"] == 0).mean()), 4),
        "two_policy_multiplier_cv_r2": round(two_policy_band_r2(df), 4),
    }


def evaluate_candidates(X_train, y_train, cv):
    fitted, rows = {}, []
    for name, (est, space) in candidates(X_train).items():
        print(f"Fitting {name} ...", flush=True)
        if space:
            search = RandomizedSearchCV(est, space, n_iter=SEARCH_ITERATIONS, scoring=SCORING,
                                        refit="mae", cv=cv, random_state=SEED, n_jobs=1)
            search.fit(X_train, y_train)
            r, i = search.cv_results_, search.best_index_
            stats = {k: (r[f"mean_test_{k}"][i], r[f"std_test_{k}"][i]) for k in SCORING}
            params = {k.split("__")[-1]: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in search.best_params_.items()}
            model = search.best_estimator_
        else:
            res = cross_validate(est, X_train, y_train, scoring=SCORING, cv=cv)
            stats = {k: (res[f"test_{k}"].mean(), res[f"test_{k}"].std()) for k in SCORING}
            params = {}
            model = clone(est).fit(X_train, y_train)
        rows.append({
            "model": name,
            "cv_mae_mean": -stats["mae"][0], "cv_mae_std": stats["mae"][1],
            "cv_r2_dollar_mean": stats["r2_dollar"][0], "cv_r2_dollar_std": stats["r2_dollar"][1],
            "cv_r2_log_mean": stats["r2_log"][0], "cv_r2_log_std": stats["r2_log"][1],
            "best_params": params,
        })
        fitted[name] = model
    return fitted, pd.DataFrame(rows)


def main():
    df = load()
    summary = {"data": structure(df)}

    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED,
        stratify=policy_tier(X["Number of Policies"]))
    tier_train = policy_tier(X_train["Number of Policies"])
    tier_test = policy_tier(X_test["Number of Policies"])
    cv = KFold(CV_FOLDS, shuffle=True, random_state=SEED)

    fitted, cv_df = evaluate_candidates(X_train, y_train, cv)
    plots.cv_comparison(cv_df)

    # One-standard-error rule over the learned models: among those whose CV MAE
    # is within one std of the best, take the simplest. The formula and the mean
    # baseline are benchmarks, not candidates.
    learned = cv_df[cv_df.model.isin(SIMPLICITY_ORDER)].set_index("model")
    top = learned["cv_mae_mean"].idxmin()
    bar = learned.loc[top, "cv_mae_mean"] + learned.loc[top, "cv_mae_std"]
    eligible = learned[learned["cv_mae_mean"] <= bar].index
    best_name = next(m for m in SIMPLICITY_ORDER if m in eligible)
    best = fitted[best_name]
    print(f"Lowest CV MAE: {top}; selected (1-SE rule): {best_name}")

    preds = {n: m.predict(X_test) for n, m in fitted.items()}
    test_rows = [{"model": n, **{f"test_{k}": v for k, v in regression_metrics(y_test, p).items()}}
                 for n, p in preds.items()]
    results = cv_df.merge(pd.DataFrame(test_rows), on="model")
    results.drop(columns="best_params").round(4).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    p_best = preds[best_name]
    tiers_best = error_by_tier(y_test, p_best, tier_test)
    tiers_formula = error_by_tier(y_test, preds["Premium x tier formula"], tier_test)
    plots.actual_vs_predicted(y_test, p_best, tier_test, best_name)
    plots.error_share(tiers_best)

    # Intervals are calibrated on out-of-fold training predictions, then checked on test.
    oof = cross_val_predict(best, X_train, y_train, cv=cv)
    conf = TierConformal(INTERVAL_COVERAGE).fit(y_train, oof, tier_train)
    lo, hi = conf.interval(p_best, tier_test)
    intervals = interval_report(y_test, lo, hi, p_best, tier_test)
    plots.interval_widths(intervals, conf.bounds_, INTERVAL_COVERAGE)

    print("Permutation importance ...", flush=True)
    pi = permutation_importance(best, X_test, y_test, scoring="neg_mean_absolute_error",
                                n_repeats=5, random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=X_test.columns).sort_values(ascending=False).head(10)
    plots.importance(imp, best_name)

    formula = fitted["Premium x tier formula"]
    summary.update({
        "split": {"train": int(len(X_train)), "test": int(len(X_test))},
        "selected_model": best_name,
        "lowest_cv_mae_model": top,
        "selected_params": cv_df.set_index("model").loc[best_name, "best_params"],
        "formula_multipliers": {k: round(v, 2) for k, v in formula.multipliers_.items()},
        "test_selected": {k: round(v, 4) for k, v in regression_metrics(y_test, p_best).items()},
        "test_formula": {k: round(v, 4) for k, v in
                         regression_metrics(y_test, preds["Premium x tier formula"]).items()},
        "error_by_tier_selected": tiers_best.round(4).to_dict("index"),
        "error_by_tier_formula": tiers_formula.round(4).to_dict("index"),
        "intervals": {"coverage_target": INTERVAL_COVERAGE,
                      "log_bounds": {k: {str(q): round(v, 4) for q, v in b.items()}
                                     for k, b in conf.bounds_.items()},
                      "test": intervals.round(4).to_dict("index")},
        "permutation_importance_mae": imp.round(1).to_dict(),
        "cv_results": cv_df.round(4).to_dict(orient="records"),
    })
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(best, ROOT / "models" / "clv_model.joblib")

    print(json.dumps({k: summary[k] for k in
                      ["selected_model", "selected_params", "formula_multipliers", "test_selected",
                       "test_formula", "error_by_tier_selected", "intervals",
                       "permutation_importance_mae"]}, indent=2, default=str))
    print(results.drop(columns="best_params").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
