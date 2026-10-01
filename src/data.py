"""Loading, cleaning and splitting the LCF 2013 teaching dataset (UKDS SN 7932)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (RAW_DATA, SEED, SEGMENT_LABELS, TARGET, TEST_SIZE)

# Columns verified identical to a kept column, or not usable (see README).
REDUNDANT = ["P550tpr", "P344pr", "A049r", "P425r"]   # duplicates of expenditure, income, hhsize, maininc
UNUSABLE = ["casenew"]                                 # random ID with repeated values, not a key
WEIGHT = "weightar"                                    # weight rescaled to mean 1
DROP_WEIGHT = ["weighta"]                              # original weight, kept only as weightar


def load_raw(path=RAW_DATA) -> pd.DataFrame:
    """Read the Stata file with numeric codes (labels are documented in the README)."""
    df = pd.read_stata(path, convert_categoricals=False)
    assert df.shape == (5144, 19), f"unexpected shape {df.shape}"
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate and unusable columns after checking they really are duplicates."""
    assert np.allclose(df["P550tpr"], df["expenditure"])
    assert np.allclose(df["P344pr"], df["income"])
    assert (df["A049r"] == df["hhsize"]).all()
    out = df.drop(columns=REDUNDANT + UNUSABLE + DROP_WEIGHT).copy()
    out["expenditure"] = out["expenditure"].astype(float)
    out["income"] = out["income"].astype(float)
    return out


def add_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Tertile segments of expenditure. Cut points are computed on the whole sample here;
    for model evaluation use make_split(), which fits the cut points on training data only."""
    out = df.copy()
    out["segment"] = pd.qcut(out[TARGET], 3, labels=SEGMENT_LABELS)
    return out


def make_split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """One shared train/test split for every model. Segment cut points come from the
    training set only, so no test information leaks into the class definition."""
    train, test = train_test_split(df, test_size=test_size, random_state=seed)
    train, test = train.copy(), test.copy()
    cuts = train[TARGET].quantile([1 / 3, 2 / 3]).to_numpy()
    for part in (train, test):
        part["segment"] = pd.cut(part[TARGET], [-np.inf, cuts[0], cuts[1], np.inf],
                                 labels=SEGMENT_LABELS)
    return train, test, cuts


def load_clean() -> pd.DataFrame:
    return clean(load_raw())
