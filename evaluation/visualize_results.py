import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "data/synthetic/telemetry_with_anomalies.csv"
MODEL_PATH = "models/telemetry_autoencoder.pth"

THRESHOLD = 2.148575

OUTPUT_DIR = "evaluation"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("ASTRA-TAD EVALUATION VISUALIZATION")
print("=" * 70)

df = pd.read_csv(DATASET_PATH)

print()
print(f"✓ Dataset loaded: {len(df)} records")


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
# SCALER VALUES
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
# AUTOENCODER
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
# CALCULATE ANOMALY SCORES
# ============================================================

X = df[FEATURES].values.astype(
    np.float32
)

X_scaled = (
    X - SCALER_MEAN
) / SCALER_SCALE


with torch.no_grad():

    tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    )

    reconstructed = model(tensor)

    errors = torch.mean(
        (tensor - reconstructed) ** 2,
        dim=1
    )

    anomaly_scores = errors.numpy()


df["anomaly_score"] = anomaly_scores

df["predicted_anomaly"] = (
    df["anomaly_score"] > THRESHOLD
).astype(int)


print("✓ Anomaly scores calculated")


# ============================================================
# GRAPH 1
# ANOMALY SCORE DISTRIBUTION
# ============================================================

plt.figure(figsize=(10, 6))

normal_scores = df[
    df["anomaly"] == 0
]["anomaly_score"]

true_anomaly_scores = df[
    df["anomaly"] == 1
]["anomaly_score"]


plt.hist(
    normal_scores,
    bins=50,
    alpha=0.7,
    label="Normal"
)

plt.hist(
    true_anomaly_scores,
    bins=50,
    alpha=0.7,
    label="Actual Anomaly"
)

plt.axvline(
    THRESHOLD,
    linestyle="--",
    linewidth=2,
    label="Anomaly Threshold"
)

plt.xlabel("Reconstruction Error")

plt.ylabel("Number of Records")

plt.title(
    "ASTRA-TAD Anomaly Score Distribution"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/anomaly_score_distribution.png",
    dpi=200
)

plt.close()

print(
    "✓ Saved anomaly_score_distribution.png"
)


# ============================================================
# GRAPH 2
# ANOMALY SCORE OVER DATASET
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    df.index,
    df["anomaly_score"],
    linewidth=0.8
)

plt.axhline(
    THRESHOLD,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

plt.xlabel("Telemetry Record")

plt.ylabel("Anomaly Score")

plt.title(
    "ASTRA-TAD Anomaly Score Over Telemetry Stream"
)

plt.legend()

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/anomaly_score_stream.png",
    dpi=200
)

plt.close()

print(
    "✓ Saved anomaly_score_stream.png"
)


# ============================================================
# GRAPH 3
# DETECTION RATE BY ANOMALY TYPE
# ============================================================

anomaly_df = df[
    df["anomaly"] == 1
]

types = []
detection_rates = []


for anomaly_type in anomaly_df[
    "anomaly_type"
].unique():

    subset = anomaly_df[
        anomaly_df["anomaly_type"]
        == anomaly_type
    ]

    total = len(subset)

    detected = np.sum(
        subset["predicted_anomaly"] == 1
    )

    rate = (
        detected / total * 100
        if total > 0
        else 0
    )

    types.append(anomaly_type)

    detection_rates.append(rate)


plt.figure(figsize=(11, 6))

plt.bar(
    types,
    detection_rates
)

plt.axhline(
    100,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Anomaly Type")

plt.ylabel("Detection Rate (%)")

plt.title(
    "ASTRA-TAD Detection Rate by Anomaly Type"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.ylim(0, 110)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/detection_rate_by_type.png",
    dpi=200
)

plt.close()

print(
    "✓ Saved detection_rate_by_type.png"
)


# ============================================================
# GRAPH 4
# AVERAGE SCORE BY ANOMALY TYPE
# ============================================================

mean_scores = []

for anomaly_type in types:

    subset = anomaly_df[
        anomaly_df["anomaly_type"]
        == anomaly_type
    ]

    mean_scores.append(
        subset["anomaly_score"].mean()
    )


plt.figure(figsize=(11, 6))

plt.bar(
    types,
    mean_scores
)

plt.axhline(
    THRESHOLD,
    linestyle="--",
    linewidth=2,
    label="Threshold"
)

plt.xlabel("Anomaly Type")

plt.ylabel("Mean Anomaly Score")

plt.title(
    "Average Anomaly Score by Anomaly Type"
)

plt.xticks(
    rotation=35,
    ha="right"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/mean_score_by_type.png",
    dpi=200
)

plt.close()

print(
    "✓ Saved mean_score_by_type.png"
)


# ============================================================
# SAVE SCORES
# ============================================================

output_csv = (
    f"{OUTPUT_DIR}/evaluation_scores.csv"
)

df.to_csv(
    output_csv,
    index=False
)

print(
    f"✓ Saved {output_csv}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("VISUALIZATION COMPLETE")
print("=" * 70)

print()
print("Generated files:")

print(
    "1. evaluation/anomaly_score_distribution.png"
)

print(
    "2. evaluation/anomaly_score_stream.png"
)

print(
    "3. evaluation/detection_rate_by_type.png"
)

print(
    "4. evaluation/mean_score_by_type.png"
)

print(
    "5. evaluation/evaluation_scores.csv"
)

print()
print("=" * 70)