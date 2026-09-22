import pandas as pd
import numpy as np
import torch
import joblib
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score


# ==========================================
# CONFIGURATION
# ==========================================

TEST_FILE = "data/processed/test_anomalies.csv"
VALIDATION_FILE = "data/processed/validation_healthy.csv"
MODEL_FILE = "models/telemetry_autoencoder.pth"
SCALER_FILE = "models/telemetry_scaler.joblib"

FEATURE_COLUMNS = [
    "temperature",
    "voltage",
    "current",
    "pressure",
    "vibration",
    "rpm",
    "battery"
]


# ==========================================
# MODEL DEFINITION
# ==========================================

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


# ==========================================
# LOAD DATA
# ==========================================

print("Loading test data...")

test_df = pd.read_csv(TEST_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

X_test = test_df[FEATURE_COLUMNS].values
X_validation = validation_df[FEATURE_COLUMNS].values

# ==========================================
# DATA IS ALREADY STANDARDIZED
# ==========================================

print("Using already-standardized telemetry data...")

# test_anomalies.csv and validation_healthy.csv
# are already standardized using the project scaler.
# Therefore, DO NOT apply StandardScaler again.


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

print("Loading trained Autoencoder...")

device = torch.device("cpu")

model = TelemetryAutoencoder(input_dim=7)

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True
    )
)

model.eval()


# ==========================================
# VALIDATION RECONSTRUCTION ERROR
# ==========================================

print("Calculating validation reconstruction errors...")

with torch.no_grad():

    validation_tensor = torch.tensor(
        X_validation,
        dtype=torch.float32
    )

    validation_output = model(validation_tensor)

    validation_errors = torch.mean(
        (validation_output - validation_tensor) ** 2,
        dim=1
    ).numpy()


# ==========================================
# THRESHOLD
# ==========================================

threshold = np.percentile(
    validation_errors,
    95
)

print(f"\n95th percentile threshold: {threshold:.6f}")


# ==========================================
# TEST RECONSTRUCTION ERROR
# ==========================================

print("Calculating test reconstruction errors...")

with torch.no_grad():

    test_tensor = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    test_output = model(test_tensor)

    test_errors = torch.mean(
        (test_output - test_tensor) ** 2,
        dim=1
    ).numpy()


# ==========================================
# PREDICTIONS
# ==========================================

predictions = (
    test_errors > threshold
).astype(int)

test_df["reconstruction_error"] = test_errors
test_df["prediction"] = predictions


# ==========================================
# OVERALL PERFORMANCE
# ==========================================

y_true = test_df["anomaly"]

precision = precision_score(
    y_true,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_true,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_true,
    predictions,
    zero_division=0
)


print("\n==============================================")
print("       OVERALL AUTOENCODER RESULTS")
print("==============================================")

print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ==========================================
# ANOMALY TYPE PERFORMANCE
# ==========================================

print("\n==============================================")
print("       ANOMALY TYPE PERFORMANCE")
print("==============================================")

results = []

anomaly_types = [
    anomaly_type
    for anomaly_type in test_df["anomaly_type"].unique()
    if anomaly_type != "normal"
]


for anomaly_type in anomaly_types:

    subset = test_df[
        test_df["anomaly_type"] == anomaly_type
    ]

    total = len(subset)

    detected = int(
        subset["prediction"].sum()
    )

    missed = total - detected

    detection_rate = (
        detected / total
        if total > 0
        else 0
    )

    average_error = (
        subset["reconstruction_error"].mean()
    )

    results.append({

        "anomaly_type": anomaly_type,

        "total": total,

        "detected": detected,

        "missed": missed,

        "detection_rate": detection_rate,

        "average_reconstruction_error":
            average_error
    })


results_df = pd.DataFrame(results)


# ==========================================
# DISPLAY
# ==========================================

print(
    results_df.to_string(
        index=False,
        formatters={
            "detection_rate":
                "{:.2%}".format,

            "average_reconstruction_error":
                "{:.4f}".format
        }
    )
)


# ==========================================
# SAVE RESULTS
# ==========================================

output_file = (
    "data/processed/anomaly_type_performance.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


print("\n==============================================")
print(
    f"Results saved to: {output_file}"
)
print("==============================================")