# Container for the FastAPI service.   Build:  docker build -t lcf-api .
FROM python:3.12-slim

# XGBoost needs the OpenMP library, which the slim image does not include.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY api ./api
COPY data/raw ./data/raw

# Train and save the two models inside the image (same code as the thesis).
RUN python -m src.model

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
