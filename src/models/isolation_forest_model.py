import os
import joblib
import pandas as pd

from sklearn.ensemble import IsolationForest


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_PATH = "data/processed/train_healthy.csv"

MODEL_DIR = "models"

MODEL_PATH = (
    f"{MODEL_DIR}/isolation_forest.joblib"
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
# CREATE MODEL DIRECTORY
# ==========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

print("Loading healthy training data...")

train_df = pd.read_csv(
    TRAIN_PATH
)

X_train = train_df[
    FEATURE_COLUMNS
]


print("Training data shape:", X_train.shape)


# ==========================================================
# CREATE ISOLATION FOREST
# ==========================================================

model = IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1
)


# ==========================================================
# TRAIN
# ==========================================================

print("\nTraining Isolation Forest...")

model.fit(X_train)


print("Training completed!")


# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    MODEL_PATH
)


print("\nModel saved to:")

print(MODEL_PATH)

print("\nIsolation Forest training completed successfully!")