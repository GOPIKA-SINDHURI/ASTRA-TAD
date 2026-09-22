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

print("Evaluation device:", device)


# ==========================================================
# LSTM AUTOENCODER
# ==========================================================

class LSTMAutoencoder(nn.Module):

    def __init__(
        self,
        input_size=7,
        hidden_size=64,
        latent_size=16
    ):

        super().__init__()

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Latent representation
        self.latent = nn.Linear(
            hidden_size,
            latent_size
        )

        # Decoder input
        self.decoder_input = nn.Linear(
            latent_size,
            hidden_size
        )

        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        # Output
        self.output_layer = nn.Linear(
            hidden_size,
            input_size
        )


    def forward(self, x):

        # Encode
        encoded_sequence, _ = (
            self.encoder(x)
        )

        # Final encoder state
        final_hidden = (
            encoded_sequence[:, -1, :]
        )

        # Latent vector
        latent = self.latent(
            final_hidden
        )

        # Prepare decoder input
        decoder_input = (
            self.decoder_input(latent)
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

        # Decode
        decoded, _ = self.decoder(
            decoder_input
        )

        # Reconstruct
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

        sequence = data[
            i:i + sequence_length
        ]

        sequences.append(sequence)

    return np.array(
        sequences,
        dtype=np.float32
    )


# ==========================================================
# CALCULATE SEQUENCE ERRORS
# ==========================================================

def calculate_errors(df, model):

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

    # Process in batches so memory usage stays reasonable

    batch_size = 128

    with torch.no_grad():

        for start in range(
            0,
            len(X),
            batch_size
        ):

            batch = X[
                start:start + batch_size
            ]

            reconstructed = model(
                batch
            )

            # MSE for each sequence
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

print("\nLoading LSTM Autoencoder...")

model = LSTMAutoencoder()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.to(device)

model.eval()

print("Model loaded successfully.")


# ==========================================================
# VALIDATION DATA
# ==========================================================

print(
    "\nLoading healthy validation data..."
)

validation_df = pd.read_csv(
    VALIDATION_PATH
)

validation_errors = (
    calculate_errors(
        validation_df,
        model
    )
)

print(
    "Validation sequences:",
    len(validation_errors)
)


# ==========================================================
# THRESHOLD
# ==========================================================

threshold = np.percentile(
    validation_errors,
    95
)


print(
    "\n95th percentile threshold:",
    threshold
)


# ==========================================================
# TEST DATA
# ==========================================================

print(
    "\nLoading test anomaly data..."
)

test_df = pd.read_csv(
    TEST_PATH
)

test_errors = (
    calculate_errors(
        test_df,
        model
    )
)


print(
    "Test sequences:",
    len(test_errors)
)


# ==========================================================
# CREATE SEQUENCE LABELS
# ==========================================================
#
# A sequence is considered anomalous if ANY
# record inside its 30-step window is anomalous.
#
# This allows us to detect an anomaly even when
# it occurs in the middle of a sequence.
#


raw_labels = test_df[
    "anomaly"
].values


sequence_labels = []


for i in range(
    len(raw_labels)
    - SEQUENCE_LENGTH
    + 1
):

    window = raw_labels[
        i:i + SEQUENCE_LENGTH
    ]

    if np.any(window == 1):

        sequence_labels.append(1)

    else:

        sequence_labels.append(0)


y_true = np.array(
    sequence_labels
)


# ==========================================================
# PREDICTIONS
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
print("       LSTM AUTOENCODER RESULTS")
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

print(
    "\nClassification Report:"
)

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
# ANOMALY STATISTICS
# ==========================================================

print(
    "\nActual anomalous sequences:",
    int(y_true.sum())
)

print(
    "Predicted anomalous sequences:",
    int(y_pred.sum())
)


print(
    "\nLSTM Autoencoder evaluation completed!"
)