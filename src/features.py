"""Feature engineering and the leak-free preprocessing step."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import policy_tier

NUMERIC = [
    "Income", "Monthly Premium Auto", "Months Since Last Claim",
    "Months Since Policy Inception", "Number of Open Complaints",
    "Number of Policies", "Total Claim Amount",
    "log_premium", "premium_to_income", "claim_per_policy", "renewal_month",
]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Row-wise features only, so nothing is learned from the test rows."""
    X = X.copy()
    X["policy_tier"] = policy_tier(X["Number of Policies"])
    X["log_premium"] = np.log(X["Monthly Premium Auto"])
    X["premium_to_income"] = X["Monthly Premium Auto"] / (X["Income"] + 1)
    X["claim_per_policy"] = X["Total Claim Amount"] / X["Number of Policies"]
    X["renewal_month"] = X["Effective To Date"].dt.month
    X["has_complaint"] = np.where(X["Number of Open Complaints"] > 0, "Yes", "No")
    X["zero_income"] = np.where(X["Income"] == 0, "Yes", "No")
    return X.drop(columns=["Effective To Date"])


def categorical_columns(X: pd.DataFrame):
    return [c for c in X.columns if c not in NUMERIC]


def build_preprocessor(X_sample: pd.DataFrame) -> ColumnTransformer:
    cats = categorical_columns(add_features(X_sample))
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore",
                                  sparse_output=False), cats),
        ],
        verbose_feature_names_out=False,
    )


def feature_step():
    return FunctionTransformer(add_features, validate=False)
