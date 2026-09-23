from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import json
from typing import Any


# ============================================================
# ASTRA-TAD FASTAPI BACKEND
# ============================================================

app = FastAPI(
    title="ASTRA-TAD API",
    description=(
        "REST API for the ASTRA-TAD real-time AI/ML "
        "telemetry anomaly detection system."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ALERTS_FILE = BASE_DIR / "alerts" / "alerts.jsonl"

DASHBOARD_FILE = BASE_DIR / "dashboard" / "index.html"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_alerts() -> list[dict[str, Any]]:
    """
    Load alerts from the JSONL alert log.
    """

    if not ALERTS_FILE.exists():
        return []

    alerts = []

    try:

        with ALERTS_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    alert = json.loads(line)

                    if isinstance(alert, dict):
                        alerts.append(alert)

                except json.JSONDecodeError:
                    continue

    except OSError:
        return []

    return alerts


def get_alert_id(alert: dict[str, Any]) -> str | None:

    return alert.get("alert_id")


def get_severity(alert: dict[str, Any]) -> str:

    return str(
        alert.get(
            "severity",
            "UNKNOWN"
        )
    ).upper()


# ============================================================
# ROOT / DASHBOARD
# ============================================================

@app.get(
    "/",
    tags=["Dashboard"],
    summary="Open ASTRA-TAD dashboard"
)
def dashboard():

    if not DASHBOARD_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="Dashboard file not found."
        )

    return FileResponse(
        DASHBOARD_FILE
    )


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health",
    tags=["System"],
    summary="Check API health"
)
def health():

    return {
        "status": "healthy",
        "service": "ASTRA-TAD API",
        "version": "1.0.0",
        "ml_system": "online",
        "alert_storage": (
            "available"
            if ALERTS_FILE.exists()
            else "waiting"
        ),
    }


# ============================================================
# SYSTEM INFORMATION
# ============================================================

@app.get(
    "/system/info",
    tags=["System"],
    summary="Get ASTRA-TAD system information"
)
def system_info():

    return {

        "project": "ASTRA-TAD",

        "full_name":
            "AI/ML Real-Time Telemetry Anomaly Detection",

        "primary_model":
            "Dense Autoencoder",

        "input_features": [
            "temperature",
            "voltage",
            "current",
            "pressure",
            "vibration",
            "rpm",
            "battery",
        ],

        "input_dimension": 7,

        "latent_dimension": 3,

        "model_architecture":
            "7 → 16 → 8 → 3 → 8 → 16 → 7",

        "anomaly_threshold": 2.148575,

        "streaming_platform":
            "Apache Kafka",

        "api_framework":
            "FastAPI",

        "alert_format":
            "JSON Lines",
    }


# ============================================================
# ALERT SUMMARY
# ============================================================

@app.get(
    "/alerts/summary",
    tags=["Alerts"],
    summary="Get anomaly alert summary"
)
def alert_summary():

    alerts = load_alerts()

    summary = {

        "total_alerts": len(alerts),

        "critical": 0,

        "high": 0,

        "medium": 0,

        "low": 0,

        "unknown": 0,
    }

    for alert in alerts:

        severity = get_severity(alert)

        if severity == "CRITICAL":
            summary["critical"] += 1

        elif severity == "HIGH":
            summary["high"] += 1

        elif severity == "MEDIUM":
            summary["medium"] += 1

        elif severity == "LOW":
            summary["low"] += 1

        else:
            summary["unknown"] += 1

    return summary


# ============================================================
# ALL ALERTS
# ============================================================

@app.get(
    "/alerts",
    tags=["Alerts"],
    summary="Get all anomaly alerts"
)
def get_alerts():

    alerts = load_alerts()

    # Newest first
    alerts.reverse()

    return {
        "count": len(alerts),
        "alerts": alerts,
    }


# ============================================================
# LATEST ALERTS
# ============================================================

@app.get(
    "/alerts/latest",
    tags=["Alerts"],
    summary="Get latest anomaly alerts"
)
def latest_alerts(
    limit: int = 10
):

    if limit < 1:
        raise HTTPException(
            status_code=400,
            detail="limit must be greater than 0."
        )

    if limit > 100:
        raise HTTPException(
            status_code=400,
            detail="limit cannot exceed 100."
        )

    alerts = load_alerts()

    alerts.reverse()

    latest = alerts[:limit]

    return {
        "count": len(latest),
        "limit": limit,
        "alerts": latest,
    }


# ============================================================
# ALERT BY ID
# ============================================================

@app.get(
    "/alerts/{alert_id}",
    tags=["Alerts"],
    summary="Get a specific anomaly alert"
)
def get_alert(
    alert_id: str
):

    alerts = load_alerts()

    for alert in alerts:

        if get_alert_id(alert) == alert_id:

            return alert

    raise HTTPException(
        status_code=404,
        detail=f"Alert '{alert_id}' not found."
    )


# ============================================================
# ANOMALY TYPES
# ============================================================

@app.get(
    "/analytics/anomaly-types",
    tags=["Analytics"],
    summary="Get anomaly type distribution"
)
def anomaly_types():

    alerts = load_alerts()

    counts = {}

    for alert in alerts:

        telemetry = alert.get(
            "telemetry",
            {}
        )

        anomaly_type = (
            telemetry.get("anomaly_type")
            or alert.get("anomaly_type")
            or alert.get("type")
            or "unknown"
        )

        anomaly_type = str(
            anomaly_type
        )

        counts[anomaly_type] = (
            counts.get(anomaly_type, 0) + 1
        )

    return {
        "total_alerts": len(alerts),
        "anomaly_types": counts,
    }


# ============================================================
# ANOMALY SCORES
# ============================================================

@app.get(
    "/analytics/scores",
    tags=["Analytics"],
    summary="Get anomaly score information"
)
def anomaly_scores():

    alerts = load_alerts()

    scores = []

    for alert in alerts:

        score = alert.get("score")

        if score is None:
            continue

        try:

            scores.append(
                float(score)
            )

        except (
            TypeError,
            ValueError
        ):
            continue

    if not scores:

        return {
            "count": 0,
            "scores": [],
        }

    return {

        "count": len(scores),

        "minimum":
            min(scores),

        "maximum":
            max(scores),

        "average":
            sum(scores) / len(scores),

        "threshold":
            2.148575,

        "scores":
            scores,
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get(
    "/api/info",
    tags=["System"],
    summary="Get available API endpoints"
)
def api_info():

    return {

        "name":
            "ASTRA-TAD REST API",

        "version":
            "1.0.0",

        "documentation":
            "/docs",

        "redoc":
            "/redoc",

        "dashboard":
            "/",

        "endpoints": [

            "/health",

            "/system/info",

            "/alerts",

            "/alerts/latest",

            "/alerts/{alert_id}",

            "/alerts/summary",

            "/analytics/anomaly-types",

            "/analytics/scores",

        ],
    }