import json
import os
from collections import deque

import joblib
import numpy as np
import torch
import torch.nn as nn
from kafka import KafkaConsumer


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"
TOPIC = "telemetry"

MODEL_PATH = "models/telemetry_autoencoder.pth"
SCALER_PATH = "models/telemetry_scaler.joblib"

AE_THRESHOLD = 2.148575

FEATURES = [
    "temperature",
    "voltage",
    "current",
    "pressure",
    "vibration",
    "rpm",
    "battery"
]

# Stuck-sensor detector settings
ROLLING_WINDOW = 20
PERSISTENCE = 10


# ============================================================
# AUTOENCODER
# ============================================================

class Autoencoder(nn.Module):

    def __init__(self, input_dim=7):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),

            nn.Linear(16, 8),
            nn.ReLU(),

            nn.Linear(8, 3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),

            nn.Linear(8, 16),
            nn.ReLU(),

            nn.Linear(16, input_dim)
        )

    def forward(self, x):

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# ============================================================
# STARTUP
# ============================================================

print("=" * 75)
print("ASTRA-TAD HYBRID REAL-TIME ANOMALY DETECTOR")
print("=" * 75)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Autoencoder not found: {MODEL_PATH}"
    )

if not os.path.exists(SCALER_PATH):

    raise FileNotFoundError(
        f"Scaler not found: {SCALER_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

model = Autoencoder(
    input_dim=len(FEATURES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("✓ Autoencoder loaded")
print(f"✓ Autoencoder threshold: {AE_THRESHOLD}")


# ============================================================
# LOAD SCALER
# ============================================================

scaler = joblib.load(
    SCALER_PATH
)

print("✓ Training scaler loaded")


# ============================================================
# STUCK SENSOR THRESHOLDS
# ============================================================
#
# These are based on the healthy validation data used earlier.
#
# The detector checks BOTH:
#
# 1. Rolling standard deviation
# 2. Rolling range
#
# A sensor is considered suspicious when both remain
# unusually low for a persistent period.
# ============================================================

STUCK_STD_THRESHOLDS = {

    "temperature": 0.458985,
    "voltage": 0.450289,
    "current": 0.454174,
    "pressure": 0.408350,
    "vibration": 0.412003,
    "rpm": 0.445717,
    "battery": 0.066155
}


STUCK_RANGE_THRESHOLDS = {

    "temperature": 1.685560,
    "voltage": 1.656759,
    "current": 1.574687,
    "pressure": 1.490320,
    "vibration": 1.416930,
    "rpm": 1.576735,
    "battery": 0.225645
}


# ============================================================
# ROLLING TELEMETRY BUFFER
# ============================================================

history = {
    feature: deque(
        maxlen=ROLLING_WINDOW
    )
    for feature in FEATURES
}


# ============================================================
# PREPROCESSING
# ============================================================

def normalize_telemetry(data):

    values = []

    for feature in FEATURES:

        if feature not in data:

            raise ValueError(
                f"Missing feature: {feature}"
            )

        values.append(
            float(data[feature])
        )

    values = np.array(
        values,
        dtype=np.float32
    ).reshape(1, -1)

    normalized = scaler.transform(
        values
    )

    return normalized[0].astype(
        np.float32
    )


# ============================================================
# AUTOENCODER DETECTION
# ============================================================

def calculate_ae_score(data):

    x = normalize_telemetry(
        data
    )

    tensor = torch.tensor(
        x,
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():

        reconstruction = model(
            tensor
        )

        error = torch.mean(
            (tensor - reconstruction) ** 2
        ).item()

    return error


# ============================================================
# UPDATE SENSOR HISTORY
# ============================================================

def update_history(data):

    for feature in FEATURES:

        history[feature].append(
            float(data[feature])
        )


# ============================================================
# STUCK SENSOR DETECTION
# ============================================================

def detect_stuck_sensor():

    if any(
        len(history[feature]) < ROLLING_WINDOW
        for feature in FEATURES
    ):

        return False, None, 0.0

    suspicious = []

    for feature in FEATURES:

        values = np.array(
            history[feature]
        )

        rolling_std = np.std(
            values
        )

        rolling_range = (
            np.max(values)
            - np.min(values)
        )

        std_limit = (
            STUCK_STD_THRESHOLDS[
                feature
            ]
        )

        range_limit = (
            STUCK_RANGE_THRESHOLDS[
                feature
            ]
        )

        # Both conditions must hold
        if (
            rolling_std <= std_limit
            and
            rolling_range <= range_limit
        ):

            suspicious.append(
                (
                    feature,
                    rolling_std,
                    rolling_range
                )
            )

    if not suspicious:

        return False, None, 0.0

    # Select the most strongly stuck sensor
    suspicious.sort(
        key=lambda item: item[1]
    )

    feature, std_value, range_value = (
        suspicious[0]
    )

    return True, feature, std_value


# ============================================================
# ROOT CAUSE
# ============================================================

def estimate_root_cause(data):

    deviations = {}

    for index, feature in enumerate(FEATURES):

        value = float(
            data[feature]
        )

        mean = scaler.mean_[index]
        scale = scaler.scale_[index]

        if scale == 0:

            z_score = 0.0

        else:

            z_score = abs(
                (value - mean)
                / scale
            )

        deviations[feature] = z_score

    root_cause = max(
        deviations,
        key=deviations.get
    )

    return (
        root_cause,
        deviations[root_cause]
    )


# ============================================================
# SEVERITY
# ============================================================

def determine_severity(
    ae_score,
    stuck_detected
):

    if ae_score >= AE_THRESHOLD * 10:

        return "CRITICAL"

    if ae_score >= AE_THRESHOLD * 5:

        return "HIGH"

    if ae_score >= AE_THRESHOLD * 2:

        return "MEDIUM"

    if stuck_detected:

        return "MEDIUM"

    return "LOW"


# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(

    TOPIC,

    bootstrap_servers=KAFKA_SERVER,

    auto_offset_reset="latest",

    enable_auto_commit=True,

    group_id="astra-hybrid-consumer",

    value_deserializer=lambda value:
        json.loads(
            value.decode("utf-8")
        )
)


print("✓ Connected to Kafka")
print(f"✓ Listening to topic: {TOPIC}")
print("✓ Stuck sensor detector enabled")
print()
print("Waiting for telemetry...")
print("=" * 75)


# ============================================================
# REAL-TIME LOOP
# ============================================================

try:

    for message in consumer:

        try:

            data = message.value

            # ------------------------------------------------
            # Update rolling history
            # ------------------------------------------------

            update_history(
                data
            )


            # ------------------------------------------------
            # Autoencoder
            # ------------------------------------------------

            ae_score = calculate_ae_score(
                data
            )

            ae_anomaly = (
                ae_score > AE_THRESHOLD
            )


            # ------------------------------------------------
            # Stuck sensor
            # ------------------------------------------------

            stuck_detected, stuck_sensor, stuck_score = (
                detect_stuck_sensor()
            )


            # ------------------------------------------------
            # HYBRID DECISION
            # ------------------------------------------------

            hybrid_anomaly = (
                ae_anomaly
                or
                stuck_detected
            )


            timestamp = data.get(
                "timestamp",
                "unknown"
            )


            # =================================================
            # ANOMALY
            # =================================================

            if hybrid_anomaly:

                severity = determine_severity(
                    ae_score,
                    stuck_detected
                )

                root_cause, root_score = (
                    estimate_root_cause(
                        data
                    )
                )


                print()
                print("🚨 HYBRID ANOMALY DETECTED")
                print("-" * 75)

                print(
                    f"Timestamp       : {timestamp}"
                )

                print(
                    f"AE Score        : "
                    f"{ae_score:.4f}"
                )

                print(
                    f"AE Threshold    : "
                    f"{AE_THRESHOLD:.4f}"
                )

                print(
                    f"AE Detection    : "
                    f"{'YES' if ae_anomaly else 'NO'}"
                )

                print(
                    f"Stuck Detection : "
                    f"{'YES' if stuck_detected else 'NO'}"
                )

                if stuck_detected:

                    print(
                        f"Stuck Sensor    : "
                        f"{stuck_sensor}"
                    )

                    print(
                        f"Stuck Score     : "
                        f"{stuck_score:.6f}"
                    )

                print(
                    f"Severity        : "
                    f"{severity}"
                )

                print(
                    f"Root Cause      : "
                    f"{root_cause}"
                )

                print(
                    f"Root Cause Score: "
                    f"{root_score:.2f}"
                )

                print(
                    f"Simulator Label : "
                    f"{data.get('anomaly_type', 'unknown')}"
                )

                print("-" * 75)


            # =================================================
            # NORMAL
            # =================================================

            else:

                print(
                    f"✓ NORMAL | "
                    f"AE={ae_score:.4f} | "
                    f"T={data['temperature']:.2f} | "
                    f"V={data['voltage']:.2f} | "
                    f"Vib={data['vibration']:.3f} | "
                    f"Battery={data['battery']:.2f}"
                )


        except Exception as error:

            print()
            print("⚠️ Error processing telemetry:")
            print(error)
            print()


# ============================================================
# SHUTDOWN
# ============================================================

except KeyboardInterrupt:

    print()
    print("Stopping hybrid ML consumer...")


finally:

    consumer.close()

    print(
        "Kafka consumer closed."
    )

    print(
        "ASTRA-TAD hybrid detector stopped."
    )