import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = (
    "data/processed/score_based_fusion_results.csv"
)

OUTPUT_FILE = (
    "data/processed/root_cause_analysis.csv"
)

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
# LOAD DATA
# ============================================================

print("\nLoading score-based detection results...")

df = pd.read_csv(INPUT_FILE)

print("Data shape:", df.shape)


# ============================================================
# VERIFY REQUIRED COLUMNS
# ============================================================

required_columns = FEATURES + [
    "score_based_prediction"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# CALCULATE HEALTHY BASELINE
# ============================================================

print("\nCalculating healthy telemetry baselines...")

healthy = df[
    df["score_based_prediction"] == 0
].copy()


if len(healthy) == 0:

    raise ValueError(
        "No healthy observations available."
    )


baseline_mean = healthy[
    FEATURES
].mean()

baseline_std = healthy[
    FEATURES
].std()


print("\nHealthy baseline:")

for feature in FEATURES:

    print(
        f"{feature:12s} "
        f"mean={baseline_mean[feature]:.4f} "
        f"std={baseline_std[feature]:.4f}"
    )


# ============================================================
# ROOT CAUSE ANALYSIS
# ============================================================

print("\nRunning Root Cause Analysis...")


anomaly_rows = df[
    df["score_based_prediction"] == 1
].copy()


results = []


for index, row in anomaly_rows.iterrows():

    contributions = {}

    # --------------------------------------------------------
    # Calculate standardized deviation
    # --------------------------------------------------------

    for feature in FEATURES:

        std = baseline_std[feature]

        if std == 0:

            deviation = 0

        else:

            deviation = abs(
                (
                    row[feature]
                    -
                    baseline_mean[feature]
                )
                / std
            )

        contributions[feature] = deviation


    # --------------------------------------------------------
    # Sort sensors by contribution
    # --------------------------------------------------------

    ranked = sorted(
        contributions.items(),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # Top root cause
    # --------------------------------------------------------

    top_sensor = ranked[0][0]

    top_score = ranked[0][1]


    # --------------------------------------------------------
    # Second contributing sensor
    # --------------------------------------------------------

    second_sensor = ranked[1][0]

    second_score = ranked[1][1]


    # --------------------------------------------------------
    # Severity
    # --------------------------------------------------------

    if top_score >= 5:

        severity = "CRITICAL"

    elif top_score >= 3:

        severity = "HIGH"

    elif top_score >= 2:

        severity = "MEDIUM"

    else:

        severity = "LOW"


    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {

        "timestamp":
            row["timestamp"],

        "anomaly_score":
            row["final_anomaly_score"],

        "root_cause_sensor":
            top_sensor,

        "root_cause_score":
            top_score,

        "second_sensor":
            second_sensor,

        "second_sensor_score":
            second_score,

        "severity":
            severity
    }


    # Add contribution of every sensor

    for feature in FEATURES:

        result[
            f"{feature}_contribution"
        ] = contributions[feature]


    results.append(result)


# ============================================================
# CREATE RESULT DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print(
    f"\nSaved RCA results to:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n")
print("=" * 65)
print("             ROOT CAUSE SUMMARY")
print("=" * 65)


print(
    f"\nTotal detected anomalies: "
    f"{len(result_df)}"
)


print("\nTop root causes:")

root_cause_counts = (
    result_df[
        "root_cause_sensor"
    ]
    .value_counts()
)


for sensor, count in root_cause_counts.items():

    percentage = (
        count
        /
        len(result_df)
        *
        100
    )

    print(
        f"{sensor:15s} "
        f"{count:4d} "
        f"({percentage:.2f}%)"
    )


# ============================================================
# SEVERITY SUMMARY
# ============================================================

print("\nSeverity distribution:")

severity_counts = (
    result_df[
        "severity"
    ]
    .value_counts()
)


for severity, count in severity_counts.items():

    print(
        f"{severity:10s}: {count}"
    )


# ============================================================
# EXAMPLE ANOMALIES
# ============================================================

print("\n")
print("=" * 65)
print("             SAMPLE RCA RESULTS")
print("=" * 65)


print(
    result_df[
        [
            "timestamp",
            "anomaly_score",
            "root_cause_sensor",
            "root_cause_score",
            "second_sensor",
            "severity"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


print("\n")
print("=" * 65)
print("Root Cause Analysis completed.")
print("=" * 65)