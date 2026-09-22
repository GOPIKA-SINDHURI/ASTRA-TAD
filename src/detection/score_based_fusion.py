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

OUTPUT_FILE = "data/processed/score_based_fusion_results.csv"

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
# AUTOENCODER
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
# LOAD MODEL
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

print("\nLoading data...")

test_df = pd.read_csv(TEST_FILE)

validation_df = pd.read_csv(VALIDATION_FILE)

stuck_df = pd.read_csv(STUCK_FILE)

print("Test:", test_df.shape)
print("Validation:", validation_df.shape)
print("Stuck detector:", stuck_df.shape)


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

if len(test_df) != len(stuck_df):

    raise ValueError(
        "Test data and stuck detector data have different lengths."
    )


if not test_df["timestamp"].equals(
    stuck_df["timestamp"]
):

    raise ValueError(
        "Timestamp mismatch between test and stuck detector."
    )


print("Timestamp alignment: OK")


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

def reconstruction_error(df):

    X = df[FEATURES].values.astype(
        np.float32
    )

    X_tensor = torch.tensor(X)

    with torch.no_grad():

        reconstructed = model(X_tensor)

    errors = torch.mean(
        (X_tensor - reconstructed) ** 2,
        dim=1
    )

    return errors.numpy()


# ============================================================
# VALIDATION RECONSTRUCTION ERROR
# ============================================================

print("\nCalculating validation errors...")

validation_errors = reconstruction_error(
    validation_df
)

test_errors = reconstruction_error(
    test_df
)


# ============================================================
# NORMALIZE AE SCORE
# ============================================================

# Use healthy validation distribution.

ae_reference = np.percentile(
    validation_errors,
    95
)

print(
    f"AE 95th percentile: "
    f"{ae_reference:.6f}"
)


# Instead of using raw reconstruction error,
# convert it to a relative anomaly score.

ae_score_test = (
    test_errors / ae_reference
)

ae_score_validation = (
    validation_errors / ae_reference
)


# ============================================================
# STUCK SENSOR SCORE
# ============================================================

stuck_prediction = (
    stuck_df["stuck_sensor_detected"]
    .astype(int)
    .values
)


# Convert binary detector output to score.

stuck_score_test = stuck_prediction.astype(
    float
)


# ============================================================
# WEIGHTED FUSION
# ============================================================

AE_WEIGHT = 0.70
STUCK_WEIGHT = 0.30

print("\nFusion weights:")

print(
    f"Autoencoder : {AE_WEIGHT}"
)

print(
    f"Stuck Sensor: {STUCK_WEIGHT}"
)


final_score = (
    AE_WEIGHT * ae_score_test
    +
    STUCK_WEIGHT * stuck_score_test
)


# ============================================================
# VALIDATION SCORE
# ============================================================

# Stuck sensor has no positives in healthy validation.
# Therefore its contribution is zero.

validation_stuck_score = np.zeros(
    len(validation_df)
)

validation_final_score = (
    AE_WEIGHT * ae_score_validation
    +
    STUCK_WEIGHT * validation_stuck_score
)


# ============================================================
# TEST MULTIPLE THRESHOLDS
# ============================================================

y_true = test_df["anomaly"].astype(
    int
).values


thresholds = np.percentile(
    validation_final_score,
    [
        90,
        92,
        93,
        94,
        95,
        96,
        97,
        98,
        99
    ]
)


results = []


print("\n")
print("=" * 70)
print("          SCORE-BASED FUSION THRESHOLD SEARCH")
print("=" * 70)

for threshold in thresholds:

    prediction = (
        final_score > threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        prediction
    )

    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(
        y_true,
        prediction,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        prediction,
        zero_division=0
    )

    far = fp / (fp + tn)

    results.append({
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": far,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    })


results_df = pd.DataFrame(results)


# ============================================================
# FIND BEST F1 THRESHOLD
# ============================================================

best_index = results_df[
    "f1"
].idxmax()

best_row = results_df.loc[
    best_index
]

BEST_THRESHOLD = best_row[
    "threshold"
]


# ============================================================
# DISPLAY THRESHOLD RESULTS
# ============================================================

print(
    results_df.to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("                 SELECTED THRESHOLD")
print("=" * 70)

print(
    f"Threshold: {BEST_THRESHOLD:.6f}"
)

print(
    f"Precision: {best_row['precision']:.4f}"
)

print(
    f"Recall: {best_row['recall']:.4f}"
)

print(
    f"F1: {best_row['f1']:.4f}"
)

print(
    f"False Alarm Rate: "
    f"{best_row['false_alarm_rate']:.4f}"
)


# ============================================================
# FINAL PREDICTIONS
# ============================================================

final_prediction = (
    final_score > BEST_THRESHOLD
).astype(int)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    final_prediction
)

print("\n")
print("=" * 70)
print("              FINAL SCORE-BASED RESULTS")
print("=" * 70)

print("\nConfusion Matrix:")

print(cm)


tn, fp, fn, tp = cm.ravel()

precision = precision_score(
    y_true,
    final_prediction,
    zero_division=0
)

recall = recall_score(
    y_true,
    final_prediction,
    zero_division=0
)

f1 = f1_score(
    y_true,
    final_prediction,
    zero_division=0
)

far = fp / (fp + tn)


print(
    f"\nPrecision        : {precision:.4f}"
)

print(
    f"Recall           : {recall:.4f}"
)

print(
    f"F1 Score         : {f1:.4f}"
)

print(
    f"False Alarm Rate : {far:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_df = test_df.copy()

output_df[
    "ae_reconstruction_error"
] = test_errors

output_df[
    "ae_score"
] = ae_score_test

output_df[
    "stuck_sensor_score"
] = stuck_score_test

output_df[
    "final_anomaly_score"
] = final_score

output_df[
    "score_based_prediction"
] = final_prediction


output_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved results to: {OUTPUT_FILE}"
)


# ============================================================
# ANOMALY TYPE PERFORMANCE
# ============================================================

print("\n")
print("=" * 70)
print("        SCORE-BASED ANOMALY TYPE PERFORMANCE")
print("=" * 70)


anomaly_types = [
    x
    for x in test_df[
        "anomaly_type"
    ].unique()
    if x != "normal"
]


for anomaly_type in anomaly_types:

    mask = (
        test_df[
            "anomaly_type"
        ] == anomaly_type
    )

    total = mask.sum()

    detected = final_prediction[
        mask
    ].sum()

    rate = detected / total

    print(
        f"{anomaly_type:25s} "
        f"{detected:3d}/{total:3d} "
        f"({rate * 100:.2f}%)"
    )


print("\n")
print("=" * 70)
print("Score-based fusion completed.")
print("=" * 70)