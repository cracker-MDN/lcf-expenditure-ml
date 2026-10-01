"""Shared constants so every notebook and the app use identical settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw" / "icfforworkbook.dta"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5

TARGET = "expenditure"

# Setting A: the four predictors of the earlier ANOVA (like-for-like with R2 = 0.412).
FEATURES_A = ["A094r", "A121r", "G018r", "G019r"]

# Setting B: all usable predictors, including income.
FEATURES_B = [
    "income", "maininc", "A172", "A093r", "A094r", "A121r",
    "SexHRP", "hhsize", "G018r", "G019r", "Gorx",
]

# Every predictor is a coded category except income, which is numeric.
NUMERIC = ["income"]
CATEGORICAL = [c for c in FEATURES_B if c not in NUMERIC]

SEGMENT_LABELS = ["Low", "Mid", "High"]
