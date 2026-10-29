"""One function that turns a household description into a prediction.
The Streamlit app (and any later API) only calls predict_household()."""
import pandas as pd

from .config import FEATURES_B

# Allowed answers for every coded question: code -> label (from the UKDS data dictionary).
CHOICES = {
    "maininc": {0: "Earnings", 1: "Other source"},
    "A172": {1: "Yes", 2: "No"},
    "A093r": {1: "Full-time working", 2: "Part-time working",
              3: "Unemployed / government training", 4: "Economically inactive"},
    "A094r": {1: "Higher managerial, administrative and professional",
              2: "Intermediate occupations", 3: "Routine and manual occupations",
              4: "Never worked, long-term unemployed, students",
              5: "Not classified"},
    "A121r": {1: "Public rented", 2: "Private rented", 3: "Owned"},
    "SexHRP": {1: "Male", 2: "Female"},
    "hhsize": {1: "1 person", 2: "2 people", 3: "3 people", 4: "4 people", 5: "5 or more people"},
    "G018r": {1: "1 adult", 2: "2 adults", 3: "3 adults", 4: "4 or more adults"},
    "G019r": {1: "No children", 2: "One child", 3: "Two or more children"},
    "Gorx": {1: "North East", 2: "North West and Merseyside", 3: "Yorkshire and the Humber",
             4: "East Midlands", 5: "West Midlands", 6: "Eastern", 7: "London",
             8: "South East", 9: "South West", 10: "Wales", 11: "Scotland",
             12: "Northern Ireland"},
}
INCOME_MIN, INCOME_MAX = 0.0, 1185.0     # weekly £; the survey income is capped at 1,184.99


def check_inputs(inputs):
    """Return a list of problems (empty list = inputs are fine)."""
    problems = []
    missing = [f for f in FEATURES_B if f not in inputs]
    if missing:
        return [f"missing inputs: {missing}"]
    for name, options in CHOICES.items():
        if inputs[name] not in options:
            problems.append(f"{name}: {inputs[name]} is not an allowed code")
    if not INCOME_MIN <= inputs["income"] <= INCOME_MAX:
        problems.append(f"income must be between {INCOME_MIN:.0f} and {INCOME_MAX:.0f} pounds per week")
    if not problems:
        # Household size must fit the number of adults and children.
        adults, kids, size = inputs["G018r"], inputs["G019r"], inputs["hhsize"]
        smallest = min(adults + {1: 0, 2: 1, 3: 2}[kids], 5)
        largest = 5 if (adults == 4 or kids == 3) else adults + {1: 0, 2: 1}[kids]
        if not smallest <= size <= largest:
            problems.append("household size does not fit the number of adults and children")
    return problems


def predict_household(inputs, regressor, classifier):
    """inputs: dict with the 11 Setting B variables. Returns amount (pounds per week),
    spending group and the group probabilities."""
    problems = check_inputs(inputs)
    if problems:
        raise ValueError("; ".join(problems))
    row = pd.DataFrame([inputs])[FEATURES_B]
    amount = float(regressor.predict(row)[0])
    group = str(classifier.predict(row)[0])
    probs = classifier.predict_proba(row)[0]
    probabilities = {str(label): float(p) for label, p in zip(classifier.classes_, probs)}
    return {"amount": amount, "group": group, "probabilities": probabilities}
