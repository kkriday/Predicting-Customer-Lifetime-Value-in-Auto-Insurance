"""Project-wide settings: paths, seed, and validation choices."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Marketing-Customer-Value-Analysis.csv"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
SEARCH_ITERATIONS = 15

TARGET = "Customer Lifetime Value"
ID_COL = "Customer"

# Prediction intervals: nominal coverage of the per-tier conformal intervals.
INTERVAL_COVERAGE = 0.80
