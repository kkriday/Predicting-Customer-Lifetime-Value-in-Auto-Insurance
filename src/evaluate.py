"""Regression metrics, per-tier error analysis, and conformal prediction intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    ape = np.abs(pred - y) / y
    return {
        "r2_dollar": r2_score(y, pred),
        "r2_log": r2_score(np.log(y), np.log(pred)),
        "mae": mean_absolute_error(y, pred),
        "rmse": mean_squared_error(y, pred) ** 0.5,
        "median_ape": float(np.median(ape)),
        "within_10pct": float((ape <= 0.10).mean()),
    }


def error_by_tier(y, pred, tier):
    err = np.abs(np.asarray(pred) - np.asarray(y))
    ape = err / np.asarray(y)
    d = pd.DataFrame({"tier": np.asarray(tier), "err": err, "ape": ape, "y": np.asarray(y)})
    out = d.groupby("tier").agg(
        customers=("err", "size"), mean_clv=("y", "mean"), mae=("err", "mean"),
        median_ape=("ape", "median"), within_10pct=("ape", lambda s: (s <= 0.10).mean()),
        abs_error=("err", "sum"),
    )
    out["share_of_customers"] = out["customers"] / out["customers"].sum()
    out["share_of_error"] = out["abs_error"] / out["abs_error"].sum()
    return out.drop(columns="abs_error")


class TierConformal:
    """Split-conformal intervals calibrated separately for each policy tier.

    Residuals are measured in log space (relative error), so an interval scales
    with the size of the prediction. Calibrating per tier gives honest, narrow
    intervals where the model is sure and wide ones where it is not.
    """

    def __init__(self, coverage):
        self.coverage = coverage

    def fit(self, y_oof, pred_oof, tier):
        r = np.log(np.asarray(y_oof)) - np.log(np.asarray(pred_oof))
        a = (1 - self.coverage) / 2
        d = pd.DataFrame({"r": r, "tier": np.asarray(tier)})
        self.bounds_ = d.groupby("tier")["r"].quantile([a, 1 - a]).unstack().to_dict("index")
        return self

    def interval(self, pred, tier):
        lo = np.array([self.bounds_[t][(1 - self.coverage) / 2] for t in tier])
        hi = np.array([self.bounds_[t][1 - (1 - self.coverage) / 2] for t in tier])
        return np.asarray(pred) * np.exp(lo), np.asarray(pred) * np.exp(hi)


def interval_report(y, lo, hi, pred, tier):
    y = np.asarray(y)
    d = pd.DataFrame({
        "tier": np.asarray(tier),
        "covered": (y >= lo) & (y <= hi),
        "rel_width": (hi - lo) / np.asarray(pred),
        "width": hi - lo,
    })
    out = d.groupby("tier").agg(coverage=("covered", "mean"),
                                median_width=("width", "median"),
                                median_rel_width=("rel_width", "median"))
    out.loc["All"] = [d["covered"].mean(), d["width"].median(), d["rel_width"].median()]
    return out
