# FastAPI service

Same prediction function as the Streamlit app (`src/predict.py`), offered as a web service.

## Run without Docker (from the project folder, with the project environment active)
    python -m src.model                      # once, creates the saved models
    python -m uvicorn api.main:app --reload
Open http://127.0.0.1:8000/docs, click POST /predict, "Try it out", paste an example and Execute.

## Run with Docker
    docker build -t lcf-api .
    docker run --rm -p 8000:8000 lcf-api
The image trains and saves the models during the build (needs `data/raw/icfforworkbook.dta`). Then open http://localhost:8000/docs.

## Example request (PowerShell)
    Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body '{"income":550,"maininc":0,"A172":1,"A093r":1,"A094r":1,"A121r":1,"SexHRP":1,"hhsize":2,"G018r":2,"G019r":1,"Gorx":1}'
Expected: amount about 454, group Mid. An impossible household (for example hhsize 1 with 2 adults) returns HTTP 422.

## Endpoints
- GET /health: {"status": "ok"}
- POST /predict: the 11 inputs (codes as in src/predict.py CHOICES) -> amount, group, probabilities
