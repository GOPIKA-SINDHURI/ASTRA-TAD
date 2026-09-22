import pandas as pd
import numpy as np


# ==========================================
# CONFIGURATION
# ==========================================

TEST_FILE = "data/processed/test_anomalies.csv"
VALIDATION_FILE = "data/processed/validation_healthy.csv"

OUTPUT_FILE = "data/processed/stuck_sensor_detection_v2.csv"

SENSOR_COLUMNS = [
    "temperature",
    "voltage",
    "current",
    "pressure",
    "vibration",
    "rpm",
    "battery"
]

# Short window used to measure local variation
WINDOW_SIZE = 20

# Longer persistence requirement
PERSISTENCE_LENGTH = 10

# Learn thresholds only from healthy validation data
THRESHOLD_PERCENTILE = 2


# ==========================================
# LOAD DATA
# ==========================================

print("Loading healthy validation data...")

validation_df = pd.read_csv(
    VALIDATION_FILE
)

print(
    f"Healthy validation records: "
    f"{len(validation_df)}"
)


print("\nLoading test data...")

test_df = pd.read_csv(
    TEST_FILE
)

print(
    f"Test records: {len(test_df)}"
)


# ==========================================
# LEARN HEALTHY THRESHOLDS
# ==========================================

print("\nLearning healthy persistence thresholds...")

thresholds = {}


for sensor in SENSOR_COLUMNS:

    rolling_std = (
        validation_df[sensor]
        .rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        )
        .std()
    )

    rolling_range = (
        validation_df[sensor]
        .rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        )
        .max()
        -
        validation_df[sensor]
        .rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        )
        .min()
    )

    valid_std = rolling_std.dropna()
    valid_range = rolling_range.dropna()

    std_threshold = np.percentile(
        valid_std,
        THRESHOLD_PERCENTILE
    )

    range_threshold = np.percentile(
        valid_range,
        THRESHOLD_PERCENTILE
    )

    thresholds[sensor] = {
        "std": std_threshold,
        "range": range_threshold
    }


# ==========================================
# DISPLAY THRESHOLDS
# ==========================================

print("\n==============================================")
print("       HEALTHY SENSOR THRESHOLDS")
print("==============================================")

for sensor in SENSOR_COLUMNS:

    print(
        f"{sensor:12s} "
        f"STD <= {thresholds[sensor]['std']:.6f}   "
        f"Range <= {thresholds[sensor]['range']:.6f}"
    )


# ==========================================
# DETECTION
# ==========================================

print("\nRunning persistence + range detection...")

sensor_detection_columns = []


for sensor in SENSOR_COLUMNS:

    rolling_std = (
        test_df[sensor]
        .rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        )
        .std()
    )

    rolling_max = (
        test_df[sensor]
        .rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        )
        .max()
    )

    rolling_min = (
        test_df[sensor]
        .rolling(
            window=WINDOW_SIZE,
            min_periods=WINDOW_SIZE
        )
        .min()
    )

    rolling_range = (
        rolling_max - rolling_min
    )

    low_variation = (
        rolling_std <= thresholds[sensor]["std"]
    )

    low_range = (
        rolling_range <= thresholds[sensor]["range"]
    )

    # Both conditions must be true
    possible_stuck = (
        low_variation &
        low_range
    )

    # Require the condition to persist
    persistent_stuck = (
        possible_stuck
        .rolling(
            window=PERSISTENCE_LENGTH,
            min_periods=PERSISTENCE_LENGTH
        )
        .sum()
        >= PERSISTENCE_LENGTH
    )

    column_name = f"{sensor}_stuck_v2"

    test_df[column_name] = (
        persistent_stuck.fillna(False)
    )

    sensor_detection_columns.append(
        column_name
    )


# ==========================================
# COMBINE SENSOR DETECTIONS
# ==========================================

test_df["stuck_sensor_detected"] = (
    test_df[sensor_detection_columns]
    .any(axis=1)
    .astype(int)
)


# ==========================================
# EVALUATION
# ==========================================

actual_stuck = (
    test_df["anomaly_type"] == "stuck_sensor"
).astype(int)

predicted_stuck = (
    test_df["stuck_sensor_detected"]
)


tp = int(
    (
        (actual_stuck == 1) &
        (predicted_stuck == 1)
    ).sum()
)

fn = int(
    (
        (actual_stuck == 1) &
        (predicted_stuck == 0)
    ).sum()
)

fp = int(
    (
        (actual_stuck == 0) &
        (predicted_stuck == 1)
    ).sum()
)

tn = int(
    (
        (actual_stuck == 0) &
        (predicted_stuck == 0)
    ).sum()
)


total_stuck = int(
    actual_stuck.sum()
)

detection_rate = (
    tp / total_stuck
    if total_stuck > 0
    else 0
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

false_alarm_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)


# ==========================================
# RESULTS
# ==========================================

print("\n==============================================")
print("      STUCK SENSOR DETECTOR V2")
print("==============================================")

print(
    f"Actual stuck-sensor records : {total_stuck}"
)

print(
    f"Detected                     : {tp}"
)

print(
    f"Missed                       : {fn}"
)

print(
    f"False alarms                 : {fp}"
)

print(
    f"Detection rate               : "
    f"{detection_rate:.2%}"
)

print(
    f"Precision                    : "
    f"{precision:.2%}"
)

print(
    f"False alarm rate             : "
    f"{false_alarm_rate:.2%}"
)


# ==========================================
# SAVE
# ==========================================

test_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================================")
print(
    f"Results saved to: {OUTPUT_FILE}"
)
print("==============================================")