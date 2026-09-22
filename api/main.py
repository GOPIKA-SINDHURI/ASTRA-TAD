import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

# ============================================================
# ASTRA-TAD FASTAPI ALERT API
# ============================================================

app = FastAPI(
    title="ASTRA-TAD Alert API",
    description="Real-time anomaly alert API for ASTRA-TAD",
    version="1.0.0"
)


# ============================================================
# CONFIGURATION
# ============================================================

ALERT_FILE = "alerts/alerts.jsonl"


# ============================================================
# HELPER — READ ALERTS
# ============================================================

def read_alerts():

    alerts = []

    if not os.path.exists(ALERT_FILE):
        return alerts

    try:

        with open(
            ALERT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:

                    alert = json.loads(line)

                    alerts.append(alert)

                except json.JSONDecodeError:

                    continue

    except OSError:

        return []

    return alerts


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": "ASTRA-TAD Alert API"
    }


# ============================================================
# GET ALL ALERTS
# ============================================================

@app.get("/alerts")
def get_alerts():

    alerts = read_alerts()

    return {
        "count": len(alerts),
        "alerts": alerts
    }


# ============================================================
# GET LATEST ALERTS
# ============================================================

@app.get("/alerts/latest")
def get_latest_alerts():

    alerts = read_alerts()

    latest = alerts[-10:]

    return {
        "count": len(latest),
        "alerts": latest
    }


# ============================================================
# GET ALERT BY ID
# ============================================================

@app.get("/alerts/{alert_id}")
def get_alert_by_id(alert_id: str):

    alerts = read_alerts()

    for alert in alerts:

        if alert.get("alert_id") == alert_id:

            return alert

    raise HTTPException(
        status_code=404,
        detail="Alert not found"
    )
@app.get("/")
def dashboard():

    return FileResponse(
        "dashboard/index.html"
    )