import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_PATH = (
    "data/processed/train_healthy.csv"
)

VALIDATION_PATH = (
    "data/processed/validation_healthy.csv"
)

TEST_PATH = (
    "data/processed/test_anomalies.csv"
)

MODEL_PATH = (
    "models/telemetry_autoencoder.pth"
)


FEATURE_COLUMNS = [
    "temperature",
    "voltage",
    "current",
    "pressure",
    "vibration",
    "rpm",
    "battery"
]


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# MODEL
# ==========================================================

class TelemetryAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(7, 16),
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
            nn.Linear(16, 7)
        )


    def forward(self, x):

        encoded = self.encoder(x)

        decoded = self.decoder(encoded)

        return decoded


# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading Autoencoder...")

model = TelemetryAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()


# ==========================================================
# FUNCTION: CALCULATE RECONSTRUCTION ERROR
# ==========================================================

def reconstruction_errors(data):

    X = data[
        FEATURE_COLUMNS
    ].values.astype(np.float32)

    X_tensor = torch.tensor(
        X
    ).to(device)

    with torch.no_grad():

        reconstructed = model(
            X_tensor
        )

    errors = torch.mean(
        (X_tensor - reconstructed) ** 2,
        dim=1
    )

    return errors.cpu().numpy()


# ==========================================================
# LOAD VALIDATION DATA
# ==========================================================

print("\nLoading healthy validation data...")

validation_df = pd.read_csv(
    VALIDATION_PATH
)


# ==========================================================
# DETERMINE THRESHOLD
# ==========================================================

validation_errors = (
    reconstruction_errors(
        validation_df
    )
)


threshold = np.percentile(
    validation_errors,
    95
)


print(
    "\n95th percentile threshold:",
    threshold
)


# ==========================================================
# LOAD TEST DATA
# ==========================================================

print("\nLoading test data...")

test_df = pd.read_csv(
    TEST_PATH
)


y_true = test_df[
    "anomaly"
].values


# ==========================================================
# CALCULATE TEST ERRORS
# ==========================================================

test_errors = (
    reconstruction_errors(
        test_df
    )
)


# ==========================================================
# CLASSIFY
# ==========================================================

y_pred = (
    test_errors > threshold
).astype(int)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n==============================================")
print("        AUTOENCODER RESULTS")
print("==============================================")

print("\nConfusion Matrix:")

print(cm)


# ==========================================================
# METRICS
# ==========================================================

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


print(
    "\nPrecision:",
    round(precision, 4)
)

print(
    "Recall   :",
    round(recall, 4)
)

print(
    "F1 Score :",
    round(f1, 4)
)


# ==========================================================
# FALSE ALARM RATE
# ==========================================================

tn, fp, fn, tp = cm.ravel()


false_alarm_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)


print(
    "False Alarm Rate:",
    round(false_alarm_rate, 4)
)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "Normal",
            "Anomaly"
        ],
        zero_division=0
    )
)


print(
    "\nAutoencoder evaluation completed!"
)