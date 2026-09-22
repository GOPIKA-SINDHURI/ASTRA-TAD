import json
import os
import uuid
from datetime import datetime, timezone


class AlertLogger:

    def __init__(self, log_file="alerts/alerts.jsonl"):

        self.log_file = log_file

        # Create directory if it doesn't exist
        directory = os.path.dirname(self.log_file)

        if directory:
            os.makedirs(directory, exist_ok=True)

    def _generate_alert_id(self):
        return "ALT-" + uuid.uuid4().hex[:6].upper()

    def create_alert(
        self,
        alert_type,
        severity,
        sensor_id,
        message,
        score=None,
        threshold=None,
        telemetry=None
    ):

        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "alert_id": self._generate_alert_id(),
            "alert_type": alert_type,
            "severity": severity,
            "sensor_id": sensor_id,
            "message": message
        }

        if score is not None:
            alert["score"] = round(float(score), 6)

        if threshold is not None:
            alert["threshold"] = round(float(threshold), 6)

        if telemetry is not None:
            alert["telemetry"] = telemetry

        return alert

    def log_alert(self, alert):

        # Write JSON object as one line
        with open(
            self.log_file,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(alert) + "\n"
            )

        # Also display on terminal
        print("\n" + "=" * 70)
        print("🚨 ASTRA-TAD ALERT")
        print("=" * 70)

        print(json.dumps(
            alert,
            indent=4
        ))

        print("=" * 70)

        return alert