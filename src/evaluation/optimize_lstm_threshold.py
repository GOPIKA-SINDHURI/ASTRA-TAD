import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)


# ==========================================================
# CONFIGURATION
# ==========================================================

VALIDATION_PATH = (
    "data/processed/validation_healthy.csv"
)

TEST_PATH = (
    "data/processed/test_anomalies.csv"
)

MODEL_PATH = (
    "models/lstm_autoencoder.pth"
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


SEQUENCE_LENGTH = 30


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================================
# MODEL
# ==========================================================

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size=7,
        hidden_size=64,
        latent_size=16
    ):

        super().__init__()

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.latent = nn.Linear(
            hidden_size,
            latent_size
        )

        self.decoder_input = nn.Linear(
            latent_size,
            hidden_size
        )

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.output_layer = nn.Linear(
            hidden_size,
            input_size
        )


    def forward(self, x):

        encoded, _ = self.encoder(x)

        final_hidden = encoded[:, -1, :]

        latent = self.latent(
            final_hidden
        )

        decoder_input = self.decoder_input(
            latent
        )

        decoder_input = (
            decoder_input
            .unsqueeze(1)
            .repeat(
                1,
                x.size(1),
                1
            )
        )

        decoded, _ = self.decoder(
            decoder_input
        )

        output = self.output_layer(
            decoded
        )

        return output


# ==========================================================
# CREATE SEQUENCES
# ==========================================================

def create_sequences(
    data,
    sequence_length
):

    sequences = []

    for i in range(
        len(data) - sequence_length + 1
    ):

        sequences.append(
            data[
                i:i + sequence_length
            ]
        )

    return np.array(
        sequences,
        dtype=np.float32
    )


# ==========================================================
# CALCULATE ERRORS
# ==========================================================

def calculate_errors(
    df,
    model
):

    values = df[
        FEATURE_COLUMNS
    ].values.astype(
        np.float32
    )

    sequences = create_sequences(
        values,
        SEQUENCE_LENGTH
    )

    X = torch.tensor(
        sequences
    ).to(device)

    errors = []

    model.eval()

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            128
        ):

            batch = X[
                start:start + 128
            ]

            reconstructed = model(
                batch
            )

            batch_errors = torch.mean(
                (
                    batch -
                    reconstructed
                ) ** 2,
                dim=(1, 2)
            )

            errors.extend(
                batch_errors
                .cpu()
                .numpy()
            )

    return np.array(errors)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("Loading LSTM Autoencoder...")

model = LSTMAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()


# ==========================================================
# VALIDATION DATA
# ==========================================================

print(
    "\nCalculating validation errors..."
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

validation_errors = calculate_errors(
    validation_df,
    model
)


# ==========================================================
# TEST DATA
# ==========================================================

print(
    "Calculating test errors..."
)

test_df = pd.read_csv(
    TEST_PATH
)

test_errors = calculate_errors(
    test_df,
    model
)


# ==========================================================
# TEST LABELS
# ==========================================================

raw_labels = test_df[
    "anomaly"
].values


y_true = []

for i in range(
    len(raw_labels)
    - SEQUENCE_LENGTH
    + 1
):

    window = raw_labels[
        i:i + SEQUENCE_LENGTH
    ]

    if np.any(window == 1):

        y_true.append(1)

    else:

        y_true.append(0)


y_true = np.array(y_true)


# ==========================================================
# THRESHOLD SEARCH
# ==========================================================

thresholds = np.percentile(
    validation_errors,
    np.arange(
        80,
        100,
        1
    )
)


results = []


print(
    "\n=============================================="
)

print(
    "        THRESHOLD OPTIMIZATION"
)

print(
    "=============================================="
)


for threshold in thresholds:

    y_pred = (
        test_errors > threshold
    ).astype(int)


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


    normal_mask = (
        y_true == 0
    )


    false_alarm_rate = (
        np.sum(
            y_pred[normal_mask] == 1
        )
        /
        np.sum(normal_mask)
    )


    results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarm_rate":
                false_alarm_rate
        }
    )


# ==========================================================
# DATAFRAME
# ==========================================================

results_df = pd.DataFrame(
    results
)


# ==========================================================
# BEST F1
# ==========================================================

best_index = (
    results_df["f1"].idxmax()
)


best = results_df.loc[
    best_index
]


print(
    "\nBEST THRESHOLD BY F1"
)

print(
    "Threshold:",
    round(
        best["threshold"],
        6
    )
)

print(
    "Precision:",
    round(
        best["precision"],
        4
    )
)

print(
    "Recall:",
    round(
        best["recall"],
        4
    )
)

print(
    "F1:",
    round(
        best["f1"],
        4
    )
)

print(
    "False Alarm Rate:",
    round(
        best["false_alarm_rate"],
        4
    )
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

results_df.to_csv(
    "data/processed/lstm_threshold_results.csv",
    index=False
)


print(
    "\nSaved threshold results to:"
)

print(
    "data/processed/lstm_threshold_results.csv"
)