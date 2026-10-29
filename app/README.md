# Streamlit tool

Run from the project folder:

    python -m src.model                 # once: trains and saves the two models into models/
    streamlit run app/streamlit_app.py  # starts the app in the browser
    pytest                              # tests (includes: saved models reproduce R2 0.568 and macro F1 0.672)

- `src/model.py`: trains and saves XGBoost (amount) and logistic regression (group), Setting B, fitted on the training split.
- `src/predict.py`: `predict_household()`, the one function the app calls (a later API can call the same one).
- `app/streamlit_app.py`: the page (form, result, notes).
