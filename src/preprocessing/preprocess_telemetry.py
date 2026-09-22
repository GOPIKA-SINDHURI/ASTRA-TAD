import os
import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ==========================================================
# CONFIGURATION
# ==========================================================

HEALTHY_PATH = "data/synthetic/healthy_telemetry.csv"
ANOMALY_PATH = "data/synthetic/telemetry_with_anomalies.csv"

PROCESSED_DIR = "data/processed"
MODEL_DIR = "models"


SENSOR_COLUMNS = [
    "temperature",
    "voltage",
    "current",
    "pressure",
    "vibration",
    "rpm",
    "battery"
]


# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ==========================================================
# LOAD DATA
# ==========================================================

print("Loading telemetry datasets...")

healthy_df = pd.read_csv(HEALTHY_PATH)
anomaly_df = pd.read_csv(ANOMALY_PATH)


print("Healthy dataset shape:", healthy_df.shape)
print("Anomaly dataset shape:", anomaly_df.shape)


# ==========================================================
# BASIC DATA VALIDATION
# ==========================================================

print("\nChecking missing values...")

print(
    healthy_df[SENSOR_COLUMNS]
    .isnull()
    .sum()
)

print("\nChecking duplicate records...")

print(
    "Healthy duplicates:",
    healthy_df.duplicated().sum()
)

print(
    "Anomaly duplicates:",
    anomaly_df.duplicated().sum()
)


# ==========================================================
# CHRONOLOGICAL TRAIN / VALIDATION SPLIT
# ==========================================================

n = len(healthy_df)

train_end = int(n * 0.70)
validation_end = int(n * 0.85)


train_df = healthy_df.iloc[:train_end].copy()

validation_df = healthy_df.iloc[
    train_end:validation_end
].copy()


print("\nData split:")
print("Training:", train_df.shape)
print("Validation:", validation_df.shape)


# ==========================================================
# FIT SCALER ONLY ON TRAINING DATA
# ==========================================================

print("\nFitting StandardScaler on healthy training data...")

scaler = StandardScaler()

scaler.fit(
    train_df[SENSOR_COLUMNS]
)


# ==========================================================
# TRANSFORM DATA
# ==========================================================

train_scaled = scaler.transform(
    train_df[SENSOR_COLUMNS]
)

validation_scaled = scaler.transform(
    validation_df[SENSOR_COLUMNS]
)

anomaly_scaled = scaler.transform(
    anomaly_df[SENSOR_COLUMNS]
)


# ==========================================================
# CREATE PROCESSED DATAFRAMES
# ==========================================================

train_processed = pd.DataFrame(
    train_scaled,
    columns=SENSOR_COLUMNS
)

validation_processed = pd.DataFrame(
    validation_scaled,
    columns=SENSOR_COLUMNS
)


anomaly_processed = pd.DataFrame(
    anomaly_scaled,
    columns=SENSOR_COLUMNS
)


# ==========================================================
# ADD TIMESTAMP
# ==========================================================

train_processed.insert(
    0,
    "timestamp",
    train_df["timestamp"].values
)

validation_processed.insert(
    0,
    "timestamp",
    validation_df["timestamp"].values
)


# ==========================================================
# ADD GROUND TRUTH TO TEST DATA
# ==========================================================

anomaly_processed.insert(
    0,
    "timestamp",
    anomaly_df["timestamp"].values
)

anomaly_processed["anomaly"] = (
    anomaly_df["anomaly"].values
)

anomaly_processed["anomaly_type"] = (
    anomaly_df["anomaly_type"].values
)


# ==========================================================
# SAVE PROCESSED DATA
# ==========================================================

train_output = (
    f"{PROCESSED_DIR}/train_healthy.csv"
)

validation_output = (
    f"{PROCESSED_DIR}/validation_healthy.csv"
)

test_output = (
    f"{PROCESSED_DIR}/test_anomalies.csv"
)


train_processed.to_csv(
    train_output,
    index=False
)

validation_processed.to_csv(
    validation_output,
    index=False
)

anomaly_processed.to_csv(
    test_output,
    index=False
)


# ==========================================================
# SAVE SCALER
# ==========================================================

scaler_path = (
    f"{MODEL_DIR}/telemetry_scaler.joblib"
)

joblib.dump(
    scaler,
    scaler_path
)


# ==========================================================
# FINAL REPORT
# ==========================================================

print("\n==============================================")
print("       ASTRA-TAD PREPROCESSING COMPLETE")
print("==============================================")

print("\nProcessed files:")

print("Training   :", train_output)
print("Validation :", validation_output)
print("Testing    :", test_output)

print("\nScaler:")
print(scaler_path)

print("\nSensor features:")

for sensor in SENSOR_COLUMNS:
    print(" -", sensor)

print("\nAnomaly distribution:")

print(
    anomaly_df["anomaly_type"]
    .value_counts()
)

print("\nPreprocessing completed successfully!")