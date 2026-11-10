# Household Expenditure Prediction (LCF 2013)

BSc thesis project, University of Dunaújváros. Machine-learning prediction and classification of weekly household
expenditure, benchmarked against an earlier ANOVA baseline (R² = 0.412), with an interactive decision-support tool.

## Data
Living Costs and Food Survey, 2013: Unrestricted Access Teaching Dataset (UK Data Service, SN 7932,
DOI 10.5255/UKDA-SN-7932-2). Contains public sector information licensed under the Open Government Licence.
5,144 households, 19 variables. Value labels are in the UKDS data dictionary.

## Setup
```
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
python -m src.model           # trains and saves the models (needed by the tests and the app)
pytest
jupyter notebook notebooks/01_eda.ipynb
```

## Design decisions (see thesis for justification)
- Target: raw weekly total expenditure (`expenditure`). No equivalence scale exists in the teaching dataset.
- Setting A: four ANOVA predictors (NS-SEC, tenure, adults, children). Setting B: all usable predictors including income.
- One shared 80/20 split (seed 42); 5-fold cross-validation on the training set.
- Main analysis unweighted; weighted run (`weightar`) as a sensitivity check.
- Classification target: expenditure tertiles, cut points fitted on the training set only.
- Dropped columns: `P550tpr`, `P344pr`, `A049r`, `P425r` (duplicates), `casenew` (not a unique key), `weighta` (kept as `weightar`).

## Layout
`src/` reusable code · `notebooks/` analysis · `models/` saved models · `app/` Streamlit · `reports/figures/` thesis figures · `tests/`
