import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# CONFIGURATION
# ============================================================

TEST_FILE = "data/processed/test_anomalies.csv"
VALIDATION_FILE = "data/processed/validation_healthy.csv"

STUCK_FILE = "data/processed/stuck_sensor_detection_v2.csv"

MODEL_FILE = "models/telemetry_autoencoder.pth"

RESULT_FILE = "data/processed/hybrid_detection_results.csv"
TYPE_RESULT_FILE = "data/processed/hybrid_anomaly_type_performance.csv"

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
# DENSE AUTOENCODER ARCHITECTURE
# ============================================================

class TelemetryAutoencoder(nn.Module):

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
# LOAD AUTOENCODER
# ============================================================

print("\nLoading Dense Autoencoder...")

model = TelemetryAutoencoder()

state_dict = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict)

model.eval()

print("Dense Autoencoder loaded.")


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading telemetry data...")

test_df = pd.read_csv(TEST_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

stuck_df = pd.read_csv(STUCK_FILE)

print("Test shape:", test_df.shape)
print("Validation shape:", validation_df.shape)
print("Stuck detector shape:", stuck_df.shape)


# ============================================================
# VERIFY DATA ALIGNMENT
# ============================================================

print("\nChecking timestamp alignment...")

if len(test_df) != len(stuck_df):

    raise ValueError(
        "ERROR: Test data and stuck detector data have different lengths."
    )

if not test_df["timestamp"].equals(stuck_df["timestamp"]):

    raise ValueError(
        "ERROR: Timestamp mismatch between test data and stuck detector."
    )

print("Timestamp alignment: OK")


# ============================================================
# FUNCTION TO CALCULATE RECONSTRUCTION ERROR
# ============================================================

def reconstruction_error(data):

    X = data[FEATURES].values.astype(np.float32)

    X_tensor = torch.tensor(X)

    with torch.no_grad():

        reconstructed = model(X_tensor)

    errors = torch.mean(
        (X_tensor - reconstructed) ** 2,
        dim=1
    )

    return errors.numpy()


# ============================================================
# CALCULATE VALIDATION THRESHOLD
# ============================================================

print("\nCalculating Autoencoder threshold...")

validation_errors = reconstruction_error(validation_df)

AE_THRESHOLD = np.percentile(
    validation_errors,
    95
)

print(
    f"Autoencoder threshold (95th percentile): "
    f"{AE_THRESHOLD:.6f}"
)


# ============================================================
# RUN AUTOENCODER ON TEST DATA
# ============================================================

print("\nRunning Dense Autoencoder on test data...")

test_errors = reconstruction_error(test_df)

ae_prediction = (
    test_errors > AE_THRESHOLD
).astype(int)

print(
    "Autoencoder anomalies detected:",
    ae_prediction.sum()
)


# ============================================================
# LOAD STUCK SENSOR V2 RESULTS
# ============================================================

print("\nLoading Stuck Sensor V2 results...")

stuck_prediction = (
    stuck_df["stuck_sensor_detected"]
    .astype(int)
    .values
)

print(
    "Stuck sensor anomalies detected:",
    stuck_prediction.sum()
)


# ============================================================
# HYBRID FUSION
# ============================================================

print("\nCombining detectors...")

# OR fusion:
#
# If either detector identifies an anomaly,
# the hybrid detector marks it as an anomaly.

hybrid_prediction = (
    (ae_prediction == 1) |
    (stuck_prediction == 1)
).astype(int)


print(
    "Hybrid anomalies detected:",
    hybrid_prediction.sum()
)


# ============================================================
# GROUND TRUTH
# ============================================================

y_true = test_df["anomaly"].astype(int).values


# ============================================================
# METRICS
# ============================================================

cm = confusion_matrix(
    y_true,
    hybrid_prediction
)

precision = precision_score(
    y_true,
    hybrid_prediction,
    zero_division=0
)

recall = recall_score(
    y_true,
    hybrid_prediction,
    zero_division=0
)

f1 = f1_score(
    y_true,
    hybrid_prediction,
    zero_division=0
)

tn, fp, fn, tp = cm.ravel()

false_alarm_rate = fp / (fp + tn)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 55)
print("             HYBRID DETECTOR RESULTS")
print("=" * 55)

print("\nConfusion Matrix:")
print(cm)

print("\nPerformance:")
print(f"Precision        : {precision:.4f}")
print(f"Recall           : {recall:.4f}")
print(f"F1 Score         : {f1:.4f}")
print(f"False Alarm Rate : {false_alarm_rate:.4f}")

print("\nDetection counts:")
print(f"True anomalies   : {y_true.sum()}")
print(f"Detected         : {hybrid_prediction.sum()}")
print(f"False positives  : {fp}")
print(f"Missed anomalies : {fn}")


# ============================================================
# SAVE ROW-LEVEL RESULTS
# ============================================================

result_df = test_df.copy()

result_df["ae_reconstruction_error"] = test_errors

result_df["ae_prediction"] = ae_prediction

result_df["stuck_sensor_prediction"] = stuck_prediction

result_df["hybrid_prediction"] = hybrid_prediction

result_df.to_csv(
    RESULT_FILE,
    index=False
)

print(
    f"\nSaved detailed results to: {RESULT_FILE}"
)


# ============================================================
# ANOMALY TYPE PERFORMANCE
# ============================================================

print("\n")
print("=" * 55)
print("       ANOMALY TYPE DETECTION PERFORMANCE")
print("=" * 55)

anomaly_types = [
    x for x in test_df["anomaly_type"].unique()
    if x != "normal"
]

type_results = []

for anomaly_type in anomaly_types:

    mask = (
        test_df["anomaly_type"] == anomaly_type
    )

    total = mask.sum()

    detected = hybrid_prediction[mask].sum()

    detection_rate = (
        detected / total
        if total > 0
        else 0
    )

    type_results.append({
        "anomaly_type": anomaly_type,
        "total": total,
        "detected": detected,
        "missed": total - detected,
        "detection_rate": detection_rate
    })

    print(
        f"{anomaly_type:25s} "
        f"{detected:3d}/{total:3d} "
        f"({detection_rate * 100:.2f}%)"
    )


type_df = pd.DataFrame(type_results)

type_df.to_csv(
    TYPE_RESULT_FILE,
    index=False
)

print(
    f"\nSaved anomaly-type results to: "
    f"{TYPE_RESULT_FILE}"
)


# ============================================================
# COMPARE AE vs HYBRID
# ============================================================

print("\n")
print("=" * 55)
print("             AE vs HYBRID")
print("=" * 55)

ae_precision = precision_score(
    y_true,
    ae_prediction,
    zero_division=0
)

ae_recall = recall_score(
    y_true,
    ae_prediction,
    zero_division=0
)

ae_f1 = f1_score(
    y_true,
    ae_prediction,
    zero_division=0
)

ae_cm = confusion_matrix(
    y_true,
    ae_prediction
)

ae_tn, ae_fp, _, _ = ae_cm.ravel()

ae_far = ae_fp / (ae_fp + ae_tn)

print("\n                    Autoencoder     Hybrid")

print(
    f"Precision           {ae_precision:.4f}         {precision:.4f}"
)

print(
    f"Recall              {ae_recall:.4f}         {recall:.4f}"
)

print(
    f"F1 Score            {ae_f1:.4f}         {f1:.4f}"
)

print(
    f"False Alarm Rate    {ae_far:.4f}         {false_alarm_rate:.4f}"
)

print("\n" + "=" * 55)
print("Hybrid detector evaluation completed.")
print("=" * 55)