import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)


# ==========================================================
# CONFIGURATION
# ==========================================================

TEST_PATH = (
    "data/processed/test_anomalies.csv"
)

MODEL_PATH = (
    "models/isolation_forest.joblib"
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
# LOAD TEST DATA
# ==========================================================

print("Loading test data...")

test_df = pd.read_csv(
    TEST_PATH
)

X_test = test_df[
    FEATURE_COLUMNS
]

y_true = test_df[
    "anomaly"
]


print("Test data shape:", X_test.shape)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading Isolation Forest...")

model = joblib.load(
    MODEL_PATH
)


# ==========================================================
# PREDICT
# ==========================================================

print("Running anomaly detection...")

predictions = model.predict(
    X_test
)


# Isolation Forest:
#
#  1  = normal
# -1  = anomaly
#
# Convert into:
#
#  0 = normal
#  1 = anomaly

y_pred = (
    predictions == -1
).astype(int)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


print("\n==============================================")
print("       ISOLATION FOREST RESULTS")
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


print("\nPrecision:", round(precision, 4))

print("Recall   :", round(recall, 4))

print("F1 Score :", round(f1, 4))


# ==========================================================
# FULL CLASSIFICATION REPORT
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


print("\nEvaluation completed successfully!")