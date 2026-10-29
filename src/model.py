"""Train, save and load the two models used by the Streamlit tool (Setting B).

Run once from the project folder:   python -m src.model
The models are fitted on the TRAINING part of the shared split, so the test scores
reported in the thesis (R2 0.568, macro F1 0.672) describe exactly these models.
"""
import json

import joblib
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from .config import CATEGORICAL, FEATURES_B, MODEL_DIR, NUMERIC, SEED, TARGET
from .data import load_clean, make_split

REGRESSOR_FILE = MODEL_DIR / "xgboost_regressor_B.joblib"
CLASSIFIER_FILE = MODEL_DIR / "logistic_classifier_B.joblib"
INFO_FILE = MODEL_DIR / "model_info.json"


def make_preprocessor(scale_income):
    """One-hot encode the coded categories; optionally scale income."""
    income_step = StandardScaler() if scale_income else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", income_step, NUMERIC),
    ])


def train_models():
    """Fit both models on the training set. Settings are the best ones found in notebooks 02 and 03."""
    df = load_clean()
    train, test, cuts = make_split(df)

    regressor = Pipeline([
        ("prep", make_preprocessor(scale_income=False)),
        ("model", XGBRegressor(n_estimators=400, max_depth=2, learning_rate=0.03,
                               random_state=SEED, n_jobs=-1)),
    ])
    regressor.fit(train[FEATURES_B], train[TARGET])

    classifier = Pipeline([
        ("prep", make_preprocessor(scale_income=True)),
        ("model", LogisticRegression(C=1, max_iter=2000)),
    ])
    classifier.fit(train[FEATURES_B], train["segment"])
    return regressor, classifier, cuts


def save_models():
    MODEL_DIR.mkdir(exist_ok=True)
    regressor, classifier, cuts = train_models()
    joblib.dump(regressor, REGRESSOR_FILE)
    joblib.dump(classifier, CLASSIFIER_FILE)
    info = {"cut_points": [float(cuts[0]), float(cuts[1])],
            "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}
    INFO_FILE.write_text(json.dumps(info, indent=2))
    print("saved:", REGRESSOR_FILE.name, CLASSIFIER_FILE.name, INFO_FILE.name)


def load_models():
    """Load the saved models (run save_models() first if the files are missing)."""
    return joblib.load(REGRESSOR_FILE), joblib.load(CLASSIFIER_FILE)


if __name__ == "__main__":
    save_models()
