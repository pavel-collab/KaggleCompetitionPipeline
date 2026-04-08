"""
Text Classifier Service with MLOps monitoring.

FastAPI service that serves the BERT classifier with:
- Prometheus metrics (latency, throughput, predictions distribution)
- Health endpoints
- Structured JSON logging for Loki
"""

import os
import time
import json
import logging
import asyncio
from datetime import datetime
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/bert_classifier")
SERVICE_NAME = os.getenv("SERVICE_NAME", "classifier")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DRIFT_DETECTOR_URL = os.getenv("DRIFT_DETECTOR_URL", "http://drift-detector:8001")

# Labels for classification
LABELS = ["CLASSIC ML", "LLM/NLP", "CV", "Other"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# ============================================================
# Structured Logging (JSON for Loki)
# ============================================================

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": SERVICE_NAME,
            "message": record.getMessage(),
            "logger": record.name,
        }

        if hasattr(record, "extra"):
            log_obj.update(record.extra)

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logging():
    """Configure structured JSON logging."""
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root_logger = logging.getLogger()
    root_logger.handlers = []
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, LOG_LEVEL))

    # Reduce noise from libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)


logger = logging.getLogger(__name__)

# ============================================================
# Prometheus Metrics
# ============================================================

# Request metrics
REQUEST_COUNT = Counter(
    "classifier_requests_total",
    "Total number of classification requests",
    ["status"]
)

REQUEST_LATENCY = Histogram(
    "classifier_request_latency_seconds",
    "Request latency in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    "classifier_predictions_total",
    "Total predictions by class",
    ["predicted_class"]
)

CONFIDENCE_HISTOGRAM = Histogram(
    "classifier_confidence_score",
    "Distribution of confidence scores",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
)

# Low confidence predictions (potential drift indicator)
LOW_CONFIDENCE_COUNT = Counter(
    "classifier_low_confidence_total",
    "Predictions with confidence below threshold",
    ["predicted_class"]
)

# Model info
MODEL_INFO = Info(
    "classifier_model",
    "Information about the loaded model"
)

# Health metrics
MODEL_LOADED = Gauge(
    "classifier_model_loaded",
    "Whether the model is loaded (1) or not (0)"
)

LAST_PREDICTION_TIME = Gauge(
    "classifier_last_prediction_timestamp",
    "Timestamp of last successful prediction"
)

# Input metrics (for drift detection)
INPUT_LENGTH_HISTOGRAM = Histogram(
    "classifier_input_length_chars",
    "Distribution of input text lengths",
    buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000]
)

# ============================================================
# Global State
# ============================================================

classifier = None
model_load_time = None
http_client: Optional[httpx.AsyncClient] = None


# ============================================================
# Drift Detector Integration
# ============================================================

async def send_to_drift_detector(
    predicted_class: str,
    confidence: float,
    input_length: int
):
    """Send prediction sample to drift detector (fire and forget)."""
    global http_client

    if http_client is None:
        return

    try:
        await http_client.post(
            f"{DRIFT_DETECTOR_URL}/sample",
            json={
                "predicted_class": predicted_class,
                "confidence": confidence,
                "input_length": input_length,
            },
            timeout=1.0,  # Short timeout, don't block main request
        )
    except Exception as e:
        # Log but don't fail the main request
        logger.debug(f"Failed to send to drift detector: {e}")

# ============================================================
# Request/Response Models
# ============================================================

class ClassifyRequest(BaseModel):
    """Classification request."""
    text: str = Field(..., min_length=1, max_length=50000, description="Text to classify")
    return_all_scores: bool = Field(False, description="Return scores for all classes")


class ClassifyResponse(BaseModel):
    """Classification response."""
    label: str
    confidence: float
    all_scores: Optional[dict] = None
    latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: str
    model_load_time: Optional[str]
    uptime_seconds: float


class BatchClassifyRequest(BaseModel):
    """Batch classification request."""
    texts: list[str] = Field(..., min_items=1, max_items=100)
    return_all_scores: bool = False


class BatchClassifyResponse(BaseModel):
    """Batch classification response."""
    results: list[ClassifyResponse]
    total_latency_ms: float

# ============================================================
# Model Loading
# ============================================================

def load_model():
    """Load the BERT classifier model."""
    global classifier, model_load_time

    logger.info(f"Loading model from {MODEL_PATH}")

    try:
        device = 0 if torch.cuda.is_available() else -1
        device_name = "cuda" if device == 0 else "cpu"

        classifier = pipeline(
            "text-classification",
            model=MODEL_PATH,
            tokenizer=MODEL_PATH,
            device=device,
            top_k=None,  # Return all scores
        )

        model_load_time = datetime.utcnow()

        MODEL_LOADED.set(1)
        MODEL_INFO.info({
            "path": MODEL_PATH,
            "device": device_name,
            "load_time": model_load_time.isoformat(),
            "labels": ",".join(LABELS),
        })

        logger.info(f"Model loaded successfully on {device_name}")

    except Exception as e:
        MODEL_LOADED.set(0)
        logger.error(f"Failed to load model: {e}")
        raise


# ============================================================
# Application Lifecycle
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    global http_client

    setup_logging()
    logger.info("Starting classifier service")

    # Create HTTP client for drift detector
    http_client = httpx.AsyncClient()
    logger.info(f"Drift detector URL: {DRIFT_DETECTOR_URL}")

    try:
        load_model()
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        # Don't raise - let the service start anyway for debugging

    yield

    # Cleanup
    if http_client:
        await http_client.aclose()

    logger.info("Shutting down classifier service")


app = FastAPI(
    title="Text Classifier Service",
    description="BERT-based text classifier with MLOps monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# ============================================================
# Middleware
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()

    response = await call_next(request)

    duration = time.time() - start_time

    # Log request (skip metrics endpoint to reduce noise)
    if request.url.path != "/metrics":
        logger.info(
            f"{request.method} {request.url.path}",
            extra={
                "extra": {
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": round(duration * 1000, 2),
                }
            }
        )

    return response

# ============================================================
# Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global model_load_time

    uptime = 0
    if model_load_time:
        uptime = (datetime.utcnow() - model_load_time).total_seconds()

    return HealthResponse(
        status="healthy" if classifier else "unhealthy",
        model_loaded=classifier is not None,
        model_path=MODEL_PATH,
        model_load_time=model_load_time.isoformat() if model_load_time else None,
        uptime_seconds=uptime,
    )


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/live")
async def liveness_check():
    """Kubernetes liveness probe."""
    return {"status": "alive"}


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.post("/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    """Classify a single text."""
    if classifier is None:
        REQUEST_COUNT.labels(status="error").inc()
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()

    try:
        # Track input characteristics
        INPUT_LENGTH_HISTOGRAM.observe(len(request.text))

        # Run inference
        results = classifier(request.text, truncation=True, max_length=512)

        # Find top prediction
        top_result = max(results[0], key=lambda x: x["score"])
        label = top_result["label"]
        confidence = top_result["score"]

        # Calculate latency
        latency = time.time() - start_time

        # Update metrics
        REQUEST_COUNT.labels(status="success").inc()
        REQUEST_LATENCY.observe(latency)
        PREDICTION_COUNT.labels(predicted_class=label).inc()
        CONFIDENCE_HISTOGRAM.observe(confidence)
        LAST_PREDICTION_TIME.set(time.time())

        # Track low confidence predictions (potential drift)
        if confidence < 0.5:
            LOW_CONFIDENCE_COUNT.labels(predicted_class=label).inc()

        # Build response
        response = ClassifyResponse(
            label=label,
            confidence=confidence,
            latency_ms=round(latency * 1000, 2),
        )

        if request.return_all_scores:
            response.all_scores = {r["label"]: r["score"] for r in results[0]}

        logger.info(
            f"Classification: {label} ({confidence:.3f})",
            extra={
                "extra": {
                    "label": label,
                    "confidence": confidence,
                    "latency_ms": response.latency_ms,
                    "input_length": len(request.text),
                }
            }
        )

        # Send to drift detector (async, non-blocking)
        asyncio.create_task(send_to_drift_detector(
            predicted_class=label,
            confidence=confidence,
            input_length=len(request.text),
        ))

        return response

    except Exception as e:
        REQUEST_COUNT.labels(status="error").inc()
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify/batch", response_model=BatchClassifyResponse)
async def classify_batch(request: BatchClassifyRequest):
    """Classify multiple texts in a batch."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    results = []

    for text in request.texts:
        single_request = ClassifyRequest(
            text=text,
            return_all_scores=request.return_all_scores
        )
        result = await classify(single_request)
        results.append(result)

    total_latency = time.time() - start_time

    return BatchClassifyResponse(
        results=results,
        total_latency_ms=round(total_latency * 1000, 2),
    )


@app.post("/reload")
async def reload_model():
    """Reload the model (useful for model updates)."""
    try:
        load_model()
        return {"status": "success", "message": "Model reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {e}")


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_path": MODEL_PATH,
        "labels": LABELS,
        "num_labels": len(LABELS),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "load_time": model_load_time.isoformat() if model_load_time else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
