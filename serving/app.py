from fastapi import FastAPI
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import joblib
import pandas as pd
import time

app = FastAPI()

model = joblib.load("model.pkl")

prediction_counter = Counter(
    "total_predictions",
    "Total prediction requests"
)

success_counter = Counter(
    "successful_predictions",
    "Successful predictions"
)

latency = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)

@app.post("/predict")
def predict(data: dict):

    prediction_counter.inc()

    start = time.time()

    df = pd.DataFrame([data])

    pred = model.predict(df)

    success_counter.inc()

    latency.observe(
        time.time() - start
    )

    return {
        "prediction": int(pred[0])
    }

@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )

@app.get("/")
def home():
    return {"status": "running"}