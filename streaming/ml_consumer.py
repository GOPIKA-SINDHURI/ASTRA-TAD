import json
import numpy as np
import torch
import torch.nn as nn
from kafka import KafkaConsumer

from streaming.alert_logger import AlertLogger


# ============================================================
# ASTRA-TAD REAL-TIME ML CONSUMER
# ============================================================

print("=" * 70)
print("ASTRA-TAD REAL-TIME ML CONSUMER")
print("=" * 70)


# ============================================================
# CONFIGURATION
# ============================================================

KAFKA_SERVER = "localhost:9092"
KAFKA_TOPIC = "telemetry"
KAFKA_GROUP = "astra-ml-consumer"

MODEL_PATH = "models/telemetry_autoencoder.pth"

ANOMALY_THRESHOLD = 2.148575


# ============================================================
# FEATURES
# IMPORTANT:
# This order MUST match the scaler/model training order.
# ============================================================

FEATURES = [
    "temperature",
    "voltage",
    "current",
    "pressure",
    "vibration",
    "rpm",
    "battery"
]


# ============================================================
# MANUAL STANDARD SCALER
#
# These values were extracted from:
# models/telemetry_scaler.joblib
#
# StandardScaler formula:
#
# normalized = (value - mean) / scale
#
# ============================================================

SCALER_MEAN = np.array(
    [
        24.9966510,
        28.0294309,
        3.00406582,
        101.015761,
        0.100708748,
        3001.14804,
        88.2520398
    ],
    dtype=np.float32
)

SCALER_SCALE = np.array(
    [
        0.757812451,
        0.289628866,
        0.145506265,
        0.475711414,
        0.0171270132,
        45.9017454,
        1.01585212
    ],
    dtype=np.float32
)


print("✓ Manual scaler parameters loaded")
print(f"✓ Scaler features: {len(SCALER_MEAN)}")


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
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

model = Autoencoder(
    input_dim=len(FEATURES)
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("✓ Autoencoder loaded")


# ============================================================
# ANOMALY THRESHOLD
# ============================================================

print(
    f"✓ Anomaly threshold: {ANOMALY_THRESHOLD}"
)


# ============================================================
# ALERT LOGGER
# ============================================================

alert_logger = AlertLogger()

print("✓ Alert logger initialized")


# ============================================================
# SEVERITY
# ============================================================

def determine_severity(score):

    if score >= ANOMALY_THRESHOLD * 10:

        return "CRITICAL"

    elif score >= ANOMALY_THRESHOLD * 5:

        return "HIGH"

    elif score >= ANOMALY_THRESHOLD * 2:

        return "MEDIUM"

    else:

        return "LOW"


# ============================================================
# ROOT CAUSE ESTIMATION
# ============================================================

def estimate_root_cause(original, reconstructed):

    errors = np.abs(
        original - reconstructed
    )

    # Get feature indices sorted by largest error
    sorted_indices = np.argsort(errors)[::-1]

    # Select the top 2 contributing features
    top_indices = sorted_indices[:2]

    root_causes = []

    for index in top_indices:

        # Only report meaningful contributors
        if errors[index] > 1.0:

            root_causes.append(
                FEATURES[index]
            )

    if len(root_causes) == 0:

        return ["Unknown"]

    return root_causes


# ============================================================
# KAFKA CONSUMER
# ============================================================

consumer = KafkaConsumer(

    KAFKA_TOPIC,

    bootstrap_servers=[KAFKA_SERVER],

    group_id=KAFKA_GROUP,

    auto_offset_reset="latest",

    enable_auto_commit=True,

    value_deserializer=lambda x: json.loads(
        x.decode("utf-8")
    )
)


print(
    f"✓ Connected to Kafka: {KAFKA_SERVER}"
)

print(
    f"✓ Listening to topic: {KAFKA_TOPIC}"
)

print("=" * 70)


# ============================================================
# REAL-TIME PROCESSING
# ============================================================

try:

    for message in consumer:

        telemetry = message.value

        print("\n" + "-" * 70)

        print("📡 TELEMETRY RECEIVED")

        print(
            json.dumps(
                telemetry,
                indent=2
            )
        )


        # ====================================================
        # SENSOR ID
        # ====================================================

        sensor_id = telemetry.get(
            "sensor_id",
            telemetry.get(
                "device_id",
                "UNKNOWN"
            )
        )


        # ====================================================
        # EXTRACT FEATURES
        # ====================================================

        try:

            values = []

            for feature in FEATURES:

                values.append(
                    float(
                        telemetry[feature]
                    )
                )

            values = np.array(
                values,
                dtype=np.float32
            )

        except (
            KeyError,
            TypeError,
            ValueError
        ) as error:

            print(
                f"⚠ Invalid telemetry data: {error}"
            )

            continue


        # ====================================================
        # MANUAL STANDARD SCALING
        #
        # Equivalent to:
        #
        # scaler.transform(
        #     values.reshape(1, -1)
        # )
        #
        # ====================================================

        try:

            normalized = (
                values - SCALER_MEAN
            ) / SCALER_SCALE

        except Exception as error:

            print(
                f"⚠ Normalization error: {error}"
            )

            continue


        # ====================================================
        # CONVERT TO TORCH TENSOR
        # ====================================================

        input_tensor = torch.tensor(
            normalized,
            dtype=torch.float32
        ).reshape(
            1,
            -1
        )


        # ====================================================
        # ML INFERENCE
        # ====================================================

        with torch.no_grad():

            reconstructed = model(
                input_tensor
            )


        # ====================================================
        # RECONSTRUCTION ERROR
        # ====================================================

        reconstruction_error = torch.mean(
            (
                input_tensor - reconstructed
            ) ** 2
        )


        anomaly_score = float(
            reconstruction_error.item()
        )


        # ====================================================
        # DETERMINE ANOMALY
        # ====================================================

        is_anomaly = (
            anomaly_score >
            ANOMALY_THRESHOLD
        )


        # ====================================================
        # ANOMALY DETECTED
        # ====================================================

        if is_anomaly:

            severity = determine_severity(
                anomaly_score
            )


            # ------------------------------------------------
            # ROOT CAUSE
            # ------------------------------------------------

            reconstructed_values = (
                reconstructed
                .cpu()
                .numpy()[0]
            )

            root_causes = estimate_root_cause(
                normalized,
                reconstructed_values
            )


            # ------------------------------------------------
            # CREATE STRUCTURED ALERT
            # ------------------------------------------------

            alert = alert_logger.create_alert(

                alert_type="ML_ANOMALY",

                severity=severity,

                sensor_id=sensor_id,

                message="Telemetry anomaly detected",

                score=anomaly_score,

                threshold=ANOMALY_THRESHOLD,

                telemetry=telemetry
            )


            # ------------------------------------------------
            # ADD ROOT CAUSE
            # ------------------------------------------------

            alert["root_causes"] = root_causes


            # ------------------------------------------------
            # LOG ALERT
            # ------------------------------------------------

            alert_logger.log_alert(
                alert
            )


            # ------------------------------------------------
            # TERMINAL OUTPUT
            # ------------------------------------------------

            print(
                f"🚨 ANOMALY DETECTED | "
                f"Severity: {severity} | "
                f"Score: {anomaly_score:.6f}"
            )

            print(
                f"🔎 Possible root cause: "
                f"{', '.join(root_causes)}"
            )


        # ====================================================
        # NORMAL TELEMETRY
        # ====================================================

        else:

            print(
                f"✓ NORMAL | "
                f"Score: {anomaly_score:.6f} | "
                f"Threshold: {ANOMALY_THRESHOLD}"
            )


# ============================================================
# STOP CONSUMER
# ============================================================

except KeyboardInterrupt:

    print("\n")
    print("=" * 70)
    print("ASTRA-TAD ML CONSUMER STOPPED")
    print("=" * 70)


finally:

    consumer.close()

    print("✓ Kafka consumer closed")