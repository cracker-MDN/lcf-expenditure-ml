"""Streamlit decision-support tool: predicts weekly household expenditure.
Run from the project folder:   streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))   # so "src" can be imported

import json

import streamlit as st

from src.model import INFO_FILE, load_models
from src.predict import CHOICES, INCOME_MAX, check_inputs, predict_household

st.set_page_config(page_title="Household expenditure estimator", layout="centered")


@st.cache_resource
def get_models():
    return load_models()


st.title("Household expenditure estimator")
st.write("Enter the characteristics of a household to get an estimated weekly expenditure "
         "and a spending group (Low, Mid or High).")

try:
    regressor, classifier = get_models()
    cut_low, cut_high = json.loads(INFO_FILE.read_text())["cut_points"]
except FileNotFoundError:
    st.error("Saved models not found. Run  python -m src.model  once, then reload this page.")
    st.stop()

st.subheader("Household characteristics")
income = st.number_input("Gross weekly household income (pounds)", min_value=0.0,
                         max_value=INCOME_MAX, value=560.0, step=10.0)
inputs = {"income": income}
labels = {
    "maininc": "Main source of income", "A172": "Internet connection at home",
    "A093r": "Economic position of the reference person",
    "A094r": "Occupational class of the reference person",
    "A121r": "Tenure", "SexHRP": "Sex of the reference person",
    "hhsize": "Household size", "G018r": "Number of adults",
    "G019r": "Number of children", "Gorx": "Region",
}
for name, text in labels.items():
    options = CHOICES[name]
    inputs[name] = st.selectbox(text, list(options), format_func=options.get,
                                index=1 if name in ("hhsize", "G018r") else 0)

problems = check_inputs(inputs)
for problem in problems:
    st.warning(problem)

if st.button("Estimate", disabled=bool(problems)):
    result = predict_household(inputs, regressor, classifier)
    st.subheader("Result")
    col1, col2 = st.columns(2)
    col1.metric("Estimated weekly expenditure", f"£{result['amount']:,.0f}")
    col2.metric("Spending group", result["group"])
    st.write("Probability of each group:")
    for group in ["Low", "Mid", "High"]:          # fixed order, one bar per group
        chance = result["probabilities"][group]
        st.write(f"{group}: {chance:.0%}")
        st.progress(chance)
    st.caption(f"Groups: Low below £{cut_low:,.0f}, Mid up to £{cut_high:,.0f}, High above.")

st.divider()
st.caption("Based on the UK Living Costs and Food Survey 2013 teaching dataset (5,144 households). "
           "Expenditure and income are capped in that dataset, so very high values are less reliable. "
           "The typical error of the amount estimate is about £190 per week. "
           "This is an estimate for study purposes, not financial advice. "
           "Contains public sector information licensed under the Open Government Licence v2.0.")
