import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = (
    "data/synthetic/"
    "telemetry_with_anomalies.csv"
)

MODEL_PATH = "models/telemetry_autoencoder.pth"

THRESHOLD = 2.148575


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ASTRA-TAD MODEL EVALUATION")
print("=" * 70)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(DATASET_PATH)

print()
print(f"✓ Dataset loaded: {len(df)} records")


# Ground truth
y_true = df["anomaly"].astype(int).values


# ============================================================
# FEATURES
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
# Same values used by ML consumer
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


# ============================================================
# AUTOENCODER MODEL
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

model = Autoencoder(input_dim=7)


model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)


model.eval()


print("✓ Autoencoder loaded")


# ============================================================
# PREPARE FEATURES
# ============================================================

X = df[FEATURES].values.astype(
    np.float32
)


# ============================================================
# NORMALIZATION
# ============================================================

X_scaled = (
    X - SCALER_MEAN
) / SCALER_SCALE


# ============================================================
# MODEL PREDICTION
# ============================================================

with torch.no_grad():

    tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    reconstructed = model(
        tensor
    )

    errors = torch.mean(
        (
            tensor - reconstructed
        ) ** 2,
        dim=1
    )

    anomaly_scores = (
        errors.numpy()
    )


# ============================================================
# APPLY ANOMALY THRESHOLD
# ============================================================

y_pred = (
    anomaly_scores > THRESHOLD
).astype(int)


# ============================================================
# MANUAL CONFUSION MATRIX
# ============================================================

tn = np.sum(
    (y_true == 0) & (y_pred == 0)
)

fp = np.sum(
    (y_true == 0) & (y_pred == 1)
)

fn = np.sum(
    (y_true == 1) & (y_pred == 0)
)

tp = np.sum(
    (y_true == 1) & (y_pred == 1)
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

total = len(y_true)


accuracy = (
    (tp + tn) / total
    if total > 0
    else 0
)


precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)


recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)


f1 = (
    2 * precision * recall
    / (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print("                 Predicted")
print("                 Normal   Anomaly")

print(
    f"Actual Normal   {tn:6d}   {fp:7d}"
)

print(
    f"Actual Anomaly  {fn:6d}   {tp:7d}"
)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

print()
print("=" * 70)
print("PERFORMANCE METRICS")
print("=" * 70)

print()

print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# PERCENTAGES
# ============================================================

print()

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# DATASET DISTRIBUTION
# ============================================================

print()
print("=" * 70)
print("DATASET DISTRIBUTION")
print("=" * 70)

print()

print(
    f"Total records      : {len(df)}"
)

print(
    f"Normal records     : {(y_true == 0).sum()}"
)

print(
    f"Anomaly records    : {(y_true == 1).sum()}"
)

print(
    f"Predicted normal   : {(y_pred == 0).sum()}"
)

print(
    f"Predicted anomaly  : {(y_pred == 1).sum()}"
)


# ============================================================
# GROUND TRUTH ANOMALY TYPES
# ============================================================

print()
print("=" * 70)
print("GROUND-TRUTH ANOMALY TYPES")
print("=" * 70)

print()

anomaly_types = (
    df[df["anomaly"] == 1]["anomaly_type"]
    .value_counts()
)


for anomaly_type, count in anomaly_types.items():

    print(
        f"{anomaly_type:<25} : {count}"
    )


# ============================================================
# PER-ANOMALY-TYPE DETECTION
# ============================================================

print()
print("=" * 70)
print("ANOMALY TYPE DETECTION")
print("=" * 70)

print()

print(
    f"{'Anomaly Type':<25}"
    f"{'Total':>10}"
    f"{'Detected':>12}"
    f"{'Detection %':>15}"
)

print("-" * 65)


for anomaly_type in anomaly_types.index:

    mask = (
        (df["anomaly"] == 1)
        &
        (df["anomaly_type"] == anomaly_type)
    )

    total_type = np.sum(mask)

    detected_type = np.sum(
        y_pred[mask] == 1
    )

    detection_rate = (
        detected_type / total_type * 100
        if total_type > 0
        else 0
    )

    print(
        f"{anomaly_type:<25}"
        f"{total_type:>10}"
        f"{detected_type:>12}"
        f"{detection_rate:>14.2f}%"
    )


# ============================================================
# SCORE STATISTICS
# ============================================================

print()
print("=" * 70)
print("ANOMALY SCORE STATISTICS")
print("=" * 70)

print()

normal_scores = anomaly_scores[
    y_true == 0
]

anomaly_scores_true = anomaly_scores[
    y_true == 1
]


print(
    f"Threshold              : {THRESHOLD:.6f}"
)

print(
    f"Normal mean score      : "
    f"{np.mean(normal_scores):.6f}"
)

print(
    f"Normal maximum score   : "
    f"{np.max(normal_scores):.6f}"
)

print(
    f"Anomaly mean score     : "
    f"{np.mean(anomaly_scores_true):.6f}"
)

print(
    f"Anomaly minimum score  : "
    f"{np.min(anomaly_scores_true):.6f}"
)

print(
    f"Anomaly maximum score  : "
    f"{np.max(anomaly_scores_true):.6f}"
)


# ============================================================
# FALSE POSITIVE / FALSE NEGATIVE RATE
# ============================================================

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)


print()
print("=" * 70)
print("ERROR RATES")
print("=" * 70)

print()

print(
    f"False Positive Rate : "
    f"{false_positive_rate * 100:.2f}%"
)

print(
    f"False Negative Rate : "
    f"{false_negative_rate * 100:.2f}%"
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)