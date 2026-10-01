from src.data import load_clean, make_split
from src.config import FEATURES_A, FEATURES_B


def test_shape_and_columns():
    df = load_clean()
    assert len(df) == 5144
    assert {"expenditure", "income", "weightar"} <= set(df.columns)
    assert set(FEATURES_A) <= set(FEATURES_B) <= set(df.columns)


def test_split_is_disjoint_and_balanced():
    df = load_clean()
    train, test, cuts = make_split(df)
    assert len(train) + len(test) == len(df)
    assert set(train.index).isdisjoint(test.index)
    assert train["segment"].value_counts(normalize=True).between(0.30, 0.37).all()
