from fastapi import FastAPI
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest

import joblib
import pandas as pd
import time
import psutil

app = FastAPI()

model = joblib.load("model.pkl")

# Counter Metrics
TOTAL_REQUESTS = Counter(
    "ml_total_requests",
    "Total requests received"
)

TOTAL_PREDICTIONS = Counter(
    "ml_total_predictions",
    "Total successful predictions"
)

TOTAL_ERRORS = Counter(
    "ml_total_errors",
    "Total prediction errors"
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Prediction latency"
)

CPU_USAGE = Gauge(
    "ml_cpu_usage_percent",
    "CPU usage percentage"
)

MEMORY_USAGE = Gauge(
    "ml_memory_usage_percent",
    "Memory usage percentage"
)

DISK_USAGE = Gauge(
    "ml_disk_usage_percent",
    "Disk usage percentage"
)

MODEL_STATUS = Gauge(
    "ml_model_status",
    "Model status"
)

ACTIVE_PROCESSES = Gauge(
    "ml_active_processes",
    "Running processes"
)

SYSTEM_UPTIME = Gauge(
    "ml_system_uptime_seconds",
    "System uptime"
)

BOOT_TIME = time.time()

MODEL_STATUS.set(1)

@app.post("/predict")
def predict(data: dict):
    TOTAL_REQUESTS.inc()
    start = time.time()
    try:
        df = pd.DataFrame([data])
        pred = model.predict(df)
        TOTAL_PREDICTIONS.inc()
        PREDICTION_LATENCY.observe(
            time.time() - start
        )
        return {
            "prediction": int(pred[0])
        }

    except Exception as e:
        TOTAL_ERRORS.inc()
        return {
            "error": str(e)
        }
        raise

@app.get("/metrics")
def metrics():
    CPU_USAGE.set(
        psutil.cpu_percent()
    )
    MEMORY_USAGE.set(
        psutil.virtual_memory().percent
    )
    DISK_USAGE.set(
        psutil.disk_usage("/").percent
    )
    ACTIVE_PROCESSES.set(
        len(psutil.pids())
    )
    SYSTEM_UPTIME.set(
        time.time() - BOOT_TIME
    )
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
    
