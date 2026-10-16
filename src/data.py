"""Loading the IBM Watson Marketing Customer Value Analysis dataset."""
import pandas as pd

from .config import DATA_PATH, ID_COL, TARGET


def load(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Effective To Date"] = pd.to_datetime(df["Effective To Date"], format="%m/%d/%y")
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    return df


def split_xy(df: pd.DataFrame):
    return df.drop(columns=[TARGET, ID_COL]), df[TARGET]


def policy_tier(n_policies: pd.Series) -> pd.Series:
    """Collapse policy count into the three tiers that actually differ in CLV.

    CLV per premium dollar is flat from 3 to 9 policies, so 3+ is one tier.
    """
    return n_policies.clip(upper=3).map({1: "1", 2: "2", 3: "3+"})
