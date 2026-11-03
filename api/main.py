"""FastAPI service: estimates weekly household expenditure.
Run from the project folder:   uvicorn api.main:app --reload
Then open http://127.0.0.1:8000/docs to try it in the browser."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.model import load_models
from src.predict import INCOME_MAX, check_inputs, predict_household

models = {}      # the two saved models, loaded once when the service starts


@asynccontextmanager
async def lifespan(app):
    models["regressor"], models["classifier"] = load_models()
    yield
    models.clear()


app = FastAPI(title="Household expenditure estimator", lifespan=lifespan)


class Household(BaseModel):
    """The 11 inputs (codes as in the UKDS data dictionary, see src/predict.py)."""
    income: float = Field(ge=0, le=INCOME_MAX, description="Gross weekly household income in pounds")
    maininc: int
    A172: int
    A093r: int
    A094r: int
    A121r: int
    SexHRP: int
    hhsize: int
    G018r: int
    G019r: int
    Gorx: int


class Prediction(BaseModel):
    amount: float
    group: str
    probabilities: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(household: Household):
    inputs = household.model_dump()
    problems = check_inputs(inputs)          # same checks as the Streamlit app
    if problems:
        raise HTTPException(status_code=422, detail=problems)
    return predict_household(inputs, models["regressor"], models["classifier"])
