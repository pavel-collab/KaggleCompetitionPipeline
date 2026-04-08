"""
Drift Detection Service for ML Classifier.

Monitors for:
- Data drift: Changes in input data distribution
- Concept drift: Changes in prediction distribution
- Model staleness: Time since last model update

Uses statistical tests to detect drift and exposes metrics for Prometheus.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional
import threading

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from starlette.responses import Response
from scipy import stats

# ============================================================
# Configuration
# ============================================================

SERVICE_NAME = os.getenv("SERVICE_NAME", "drift-detector")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Drift detection parameters
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "1000"))  # Number of samples to track
REFERENCE_WINDOW_SIZE = int(os.getenv("REFERENCE_WINDOW_SIZE", "5000"))  # Reference distribution
DRIFT_CHECK_INTERVAL = int(os.getenv("DRIFT_CHECK_INTERVAL", "300"))  # Check every 5 minutes
PSI_THRESHOLD = float(os.getenv("PSI_THRESHOLD", "0.2"))  # Population Stability Index threshold
KS_THRESHOLD = float(os.getenv("KS_THRESHOLD", "0.05"))  # Kolmogorov-Smirnov p-value threshold

# Labels for classification
LABELS = ["CLASSIC ML", "LLM/NLP", "CV", "Other"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

# ============================================================
# Logging
# ============================================================

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": SERVICE_NAME,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logging.basicConfig(level=getattr(logging, LOG_LEVEL), handlers=[handler])
logger = logging.getLogger(__name__)

# ============================================================
# Prometheus Metrics
# ============================================================

# Drift indicators
DATA_DRIFT_DETECTED = Gauge(
    "drift_data_drift_detected",
    "Whether data drift is detected (1) or not (0)"
)

CONCEPT_DRIFT_DETECTED = Gauge(
    "drift_concept_drift_detected",
    "Whether concept drift is detected (1) or not (0)"
)

# Drift scores
PSI_SCORE = Gauge(
    "drift_psi_score",
    "Population Stability Index score",
    ["feature"]
)

KS_STATISTIC = Gauge(
    "drift_ks_statistic",
    "Kolmogorov-Smirnov test statistic",
    ["feature"]
)

KS_PVALUE = Gauge(
    "drift_ks_pvalue",
    "Kolmogorov-Smirnov test p-value",
    ["feature"]
)

# Distribution metrics
CLASS_DISTRIBUTION_CURRENT = Gauge(
    "drift_class_distribution_current",
    "Current class distribution percentage",
    ["class_name"]
)

CLASS_DISTRIBUTION_REFERENCE = Gauge(
    "drift_class_distribution_reference",
    "Reference class distribution percentage",
    ["class_name"]
)

# Sample counts
SAMPLES_RECEIVED = Counter(
    "drift_samples_received_total",
    "Total number of samples received for drift detection"
)

CURRENT_WINDOW_SIZE = Gauge(
    "drift_current_window_size",
    "Number of samples in current window"
)

REFERENCE_WINDOW_SIZE_GAUGE = Gauge(
    "drift_reference_window_size",
    "Number of samples in reference window"
)

# Model age
MODEL_AGE_SECONDS = Gauge(
    "drift_model_age_seconds",
    "Time since model was last updated"
)

# Service info
DRIFT_DETECTOR_INFO = Info(
    "drift_detector",
    "Information about drift detector configuration"
)

# ============================================================
# Data Structures
# ============================================================

class DriftDetector:
    """Drift detection using statistical tests."""

    def __init__(self, window_size: int, reference_size: int):
        self.window_size = window_size
        self.reference_size = reference_size

        # Sliding windows
        self.current_confidences = deque(maxlen=window_size)
        self.current_input_lengths = deque(maxlen=window_size)
        self.current_predictions = deque(maxlen=window_size)

        # Reference distributions (larger window)
        self.reference_confidences = deque(maxlen=reference_size)
        self.reference_input_lengths = deque(maxlen=reference_size)
        self.reference_predictions = deque(maxlen=reference_size)

        # Model metadata
        self.model_load_time = datetime.utcnow()

        # Lock for thread safety
        self.lock = threading.Lock()

        logger.info(f"DriftDetector initialized with window={window_size}, reference={reference_size}")

    def add_sample(
        self,
        predicted_class: str,
        confidence: float,
        input_length: int
    ):
        """Add a new prediction sample."""
        with self.lock:
            label_id = LABEL2ID.get(predicted_class, -1)

            self.current_confidences.append(confidence)
            self.current_input_lengths.append(input_length)
            self.current_predictions.append(label_id)

            # Also add to reference (it has larger window)
            self.reference_confidences.append(confidence)
            self.reference_input_lengths.append(input_length)
            self.reference_predictions.append(label_id)

            SAMPLES_RECEIVED.inc()
            CURRENT_WINDOW_SIZE.set(len(self.current_confidences))
            REFERENCE_WINDOW_SIZE_GAUGE.set(len(self.reference_confidences))

    def calculate_psi(self, reference: list, current: list, bins: int = 10) -> float:
        """Calculate Population Stability Index."""
        if len(reference) < 100 or len(current) < 100:
            return 0.0

        try:
            # Create bins from reference data
            min_val = min(min(reference), min(current))
            max_val = max(max(reference), max(current))

            if min_val == max_val:
                return 0.0

            bin_edges = np.linspace(min_val, max_val, bins + 1)

            # Calculate distributions
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            cur_counts, _ = np.histogram(current, bins=bin_edges)

            # Normalize to percentages (avoid zeros)
            ref_pct = (ref_counts + 1) / (len(reference) + bins)
            cur_pct = (cur_counts + 1) / (len(current) + bins)

            # Calculate PSI
            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return float(psi)

        except Exception as e:
            logger.error(f"PSI calculation error: {e}")
            return 0.0

    def calculate_ks_test(self, reference: list, current: list) -> tuple:
        """Perform Kolmogorov-Smirnov test."""
        if len(reference) < 100 or len(current) < 100:
            return 0.0, 1.0

        try:
            statistic, pvalue = stats.ks_2samp(reference, current)
            return float(statistic), float(pvalue)
        except Exception as e:
            logger.error(f"KS test error: {e}")
            return 0.0, 1.0

    def calculate_class_distribution(self, predictions: list) -> dict:
        """Calculate class distribution percentages."""
        if not predictions:
            return {label: 0.0 for label in LABELS}

        total = len(predictions)
        distribution = {}

        for i, label in enumerate(LABELS):
            count = predictions.count(i)
            distribution[label] = count / total

        return distribution

    def check_drift(self) -> dict:
        """Perform drift detection checks."""
        with self.lock:
            # Need enough samples
            if len(self.current_confidences) < 100:
                return {
                    "status": "insufficient_data",
                    "current_samples": len(self.current_confidences),
                    "data_drift": False,
                    "concept_drift": False,
                }

            # Split reference into first half for comparison
            ref_half = len(self.reference_confidences) // 2
            if ref_half < 100:
                ref_confidences = list(self.reference_confidences)
                ref_lengths = list(self.reference_input_lengths)
                ref_predictions = list(self.reference_predictions)
            else:
                ref_confidences = list(self.reference_confidences)[:ref_half]
                ref_lengths = list(self.reference_input_lengths)[:ref_half]
                ref_predictions = list(self.reference_predictions)[:ref_half]

            cur_confidences = list(self.current_confidences)
            cur_lengths = list(self.current_input_lengths)
            cur_predictions = list(self.current_predictions)

            # Calculate metrics
            results = {}

            # Confidence PSI
            conf_psi = self.calculate_psi(ref_confidences, cur_confidences)
            PSI_SCORE.labels(feature="confidence").set(conf_psi)
            results["confidence_psi"] = conf_psi

            # Input length PSI
            length_psi = self.calculate_psi(ref_lengths, cur_lengths)
            PSI_SCORE.labels(feature="input_length").set(length_psi)
            results["input_length_psi"] = length_psi

            # Confidence KS test
            conf_ks_stat, conf_ks_pval = self.calculate_ks_test(ref_confidences, cur_confidences)
            KS_STATISTIC.labels(feature="confidence").set(conf_ks_stat)
            KS_PVALUE.labels(feature="confidence").set(conf_ks_pval)
            results["confidence_ks_statistic"] = conf_ks_stat
            results["confidence_ks_pvalue"] = conf_ks_pval

            # Input length KS test
            length_ks_stat, length_ks_pval = self.calculate_ks_test(ref_lengths, cur_lengths)
            KS_STATISTIC.labels(feature="input_length").set(length_ks_stat)
            KS_PVALUE.labels(feature="input_length").set(length_ks_pval)
            results["input_length_ks_statistic"] = length_ks_stat
            results["input_length_ks_pvalue"] = length_ks_pval

            # Class distributions
            ref_dist = self.calculate_class_distribution(ref_predictions)
            cur_dist = self.calculate_class_distribution(cur_predictions)

            for label in LABELS:
                CLASS_DISTRIBUTION_REFERENCE.labels(class_name=label).set(ref_dist[label])
                CLASS_DISTRIBUTION_CURRENT.labels(class_name=label).set(cur_dist[label])

            results["reference_distribution"] = ref_dist
            results["current_distribution"] = cur_dist

            # Detect data drift (input characteristics changed)
            data_drift = (
                conf_psi > PSI_THRESHOLD or
                length_psi > PSI_THRESHOLD or
                conf_ks_pval < KS_THRESHOLD or
                length_ks_pval < KS_THRESHOLD
            )
            DATA_DRIFT_DETECTED.set(1 if data_drift else 0)
            results["data_drift"] = data_drift

            # Detect concept drift (prediction distribution changed)
            # Calculate Jensen-Shannon divergence for class distribution
            ref_probs = np.array([ref_dist[l] for l in LABELS])
            cur_probs = np.array([cur_dist[l] for l in LABELS])

            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            ref_probs = ref_probs + epsilon
            cur_probs = cur_probs + epsilon
            ref_probs = ref_probs / ref_probs.sum()
            cur_probs = cur_probs / cur_probs.sum()

            # JS divergence
            m = (ref_probs + cur_probs) / 2
            js_div = 0.5 * (stats.entropy(ref_probs, m) + stats.entropy(cur_probs, m))
            results["class_js_divergence"] = float(js_div)

            concept_drift = js_div > 0.1  # Threshold for JS divergence
            CONCEPT_DRIFT_DETECTED.set(1 if concept_drift else 0)
            results["concept_drift"] = concept_drift

            # Model age
            model_age = (datetime.utcnow() - self.model_load_time).total_seconds()
            MODEL_AGE_SECONDS.set(model_age)
            results["model_age_seconds"] = model_age

            results["status"] = "ok"
            results["current_samples"] = len(self.current_confidences)
            results["reference_samples"] = len(self.reference_confidences)

            return results

    def reset_model_time(self):
        """Reset model load time (call when model is reloaded)."""
        self.model_load_time = datetime.utcnow()
        logger.info("Model time reset")


# Global detector instance
detector = DriftDetector(WINDOW_SIZE, REFERENCE_WINDOW_SIZE)

# ============================================================
# Request/Response Models
# ============================================================

class PredictionSample(BaseModel):
    """Sample from classifier for drift detection."""
    predicted_class: str
    confidence: float = Field(..., ge=0, le=1)
    input_length: int = Field(..., ge=1)


class BatchSamples(BaseModel):
    """Batch of prediction samples."""
    samples: list[PredictionSample]


class DriftCheckResponse(BaseModel):
    """Response from drift check."""
    status: str
    current_samples: int
    reference_samples: Optional[int] = None
    data_drift: bool
    concept_drift: bool
    confidence_psi: Optional[float] = None
    input_length_psi: Optional[float] = None
    class_js_divergence: Optional[float] = None
    model_age_seconds: Optional[float] = None


# ============================================================
# Background Drift Checker
# ============================================================

def background_drift_checker():
    """Background thread that periodically checks for drift."""
    while True:
        try:
            results = detector.check_drift()
            if results["status"] == "ok":
                logger.info(
                    f"Drift check: data_drift={results['data_drift']}, "
                    f"concept_drift={results['concept_drift']}, "
                    f"samples={results['current_samples']}"
                )
        except Exception as e:
            logger.error(f"Background drift check error: {e}")

        time.sleep(DRIFT_CHECK_INTERVAL)


# ============================================================
# FastAPI Application
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("Starting drift detector service")

    DRIFT_DETECTOR_INFO.info({
        "window_size": str(WINDOW_SIZE),
        "reference_size": str(REFERENCE_WINDOW_SIZE),
        "psi_threshold": str(PSI_THRESHOLD),
        "ks_threshold": str(KS_THRESHOLD),
        "check_interval": str(DRIFT_CHECK_INTERVAL),
    })

    # Start background checker
    checker_thread = threading.Thread(target=background_drift_checker, daemon=True)
    checker_thread.start()
    logger.info("Background drift checker started")

    yield

    logger.info("Shutting down drift detector service")


app = FastAPI(
    title="Drift Detection Service",
    description="Monitors for data and concept drift in ML classifier",
    version="1.0.0",
    lifespan=lifespan,
)

# ============================================================
# Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": SERVICE_NAME,
        "current_samples": len(detector.current_confidences),
        "reference_samples": len(detector.reference_confidences),
    }


@app.get("/ready")
async def readiness_check():
    """Kubernetes readiness probe."""
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


@app.post("/sample")
async def add_sample(sample: PredictionSample):
    """Add a single prediction sample."""
    detector.add_sample(
        predicted_class=sample.predicted_class,
        confidence=sample.confidence,
        input_length=sample.input_length,
    )
    return {"status": "ok"}


@app.post("/samples")
async def add_samples(batch: BatchSamples):
    """Add multiple prediction samples."""
    for sample in batch.samples:
        detector.add_sample(
            predicted_class=sample.predicted_class,
            confidence=sample.confidence,
            input_length=sample.input_length,
        )
    return {"status": "ok", "count": len(batch.samples)}


@app.get("/drift", response_model=DriftCheckResponse)
async def check_drift():
    """Perform drift detection check."""
    results = detector.check_drift()
    return DriftCheckResponse(**results)


@app.post("/reset-reference")
async def reset_reference():
    """Reset reference distribution (use after model retraining)."""
    global detector
    detector = DriftDetector(WINDOW_SIZE, REFERENCE_WINDOW_SIZE)
    return {"status": "ok", "message": "Reference distribution reset"}


@app.post("/model-updated")
async def model_updated():
    """Notify that model has been updated."""
    detector.reset_model_time()
    return {"status": "ok", "message": "Model time reset"}


@app.get("/distributions")
async def get_distributions():
    """Get current and reference class distributions."""
    with detector.lock:
        ref_dist = detector.calculate_class_distribution(
            list(detector.reference_predictions)
        )
        cur_dist = detector.calculate_class_distribution(
            list(detector.current_predictions)
        )

    return {
        "reference": ref_dist,
        "current": cur_dist,
        "reference_samples": len(detector.reference_predictions),
        "current_samples": len(detector.current_predictions),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
