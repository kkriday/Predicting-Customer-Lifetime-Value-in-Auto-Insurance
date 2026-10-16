"""Candidate models. All predict CLV in dollars.

Learned models fit log(CLV) internally (TransformedTargetRegressor), which
tames the right skew, and their predictions are mapped back to dollars so every
model is scored on the same business scale.
"""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .config import SEED
from .data import policy_tier
from .features import build_preprocessor, feature_step


class PremiumTierFormula(BaseEstimator, RegressorMixin):
    """CLV = monthly premium x a multiplier learned per policy tier (1, 2, 3+).

    Two inputs, three learned numbers. It is the interpretable benchmark every
    ML model in this project has to beat.
    """

    def fit(self, X, y):
        tier = policy_tier(X["Number of Policies"])
        ratio = np.asarray(y) / X["Monthly Premium Auto"].to_numpy()
        self.multipliers_ = pd.Series(ratio, index=tier.index).groupby(tier).median().to_dict()
        return self

    def predict(self, X):
        tier = policy_tier(X["Number of Policies"])
        return X["Monthly Premium Auto"].to_numpy() * tier.map(self.multipliers_).to_numpy()


def log_target(pipeline):
    return TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)


def make_pipeline(X_sample, estimator):
    return log_target(Pipeline([
        ("features", feature_step()),
        ("prep", build_preprocessor(X_sample)),
        ("model", estimator),
    ]))


def candidates(X_sample):
    """name -> (estimator, search space). An empty space means no tuning."""
    p = "regressor__model__"
    return {
        "Baseline (mean CLV)": (DummyRegressor(strategy="mean"), {}),
        "Premium x tier formula": (PremiumTierFormula(), {}),
        "Ridge Regression": (
            make_pipeline(X_sample, Ridge()),
            {f"{p}alpha": loguniform(1e-2, 1e2)},
        ),
        "Random Forest": (
            make_pipeline(X_sample, RandomForestRegressor(n_jobs=-1, random_state=SEED)),
            {
                f"{p}n_estimators": randint(200, 500),
                f"{p}max_depth": [None, 12, 20, 30],
                f"{p}min_samples_leaf": randint(1, 10),
                f"{p}max_features": uniform(0.3, 0.7),
            },
        ),
        "XGBoost": (
            make_pipeline(X_sample, XGBRegressor(n_jobs=-1, random_state=SEED, verbosity=0)),
            {
                f"{p}n_estimators": randint(200, 800),
                f"{p}learning_rate": loguniform(0.01, 0.2),
                f"{p}max_depth": randint(3, 9),
                f"{p}subsample": uniform(0.6, 0.4),
                f"{p}colsample_bytree": uniform(0.5, 0.5),
                f"{p}min_child_weight": randint(1, 20),
            },
        ),
        "LightGBM": (
            make_pipeline(X_sample, LGBMRegressor(random_state=SEED, verbose=-1)),
            {
                f"{p}n_estimators": randint(200, 800),
                f"{p}learning_rate": loguniform(0.01, 0.2),
                f"{p}num_leaves": randint(8, 96),
                f"{p}min_child_samples": randint(5, 60),
                f"{p}subsample": uniform(0.6, 0.4),
                f"{p}subsample_freq": [1],
                f"{p}colsample_bytree": uniform(0.5, 0.5),
            },
        ),
    }
