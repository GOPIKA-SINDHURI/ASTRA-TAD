import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ==========================================================
# CONFIGURATION
# ==========================================================

TRAIN_PATH = "data/processed/train_healthy.csv"

MODEL_DIR = "models"

MODEL_PATH = (
    f"{MODEL_DIR}/lstm_autoencoder.pth"
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

BATCH_SIZE = 64

EPOCHS = 30

LEARNING_RATE = 0.001

RANDOM_SEED = 42


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

torch.manual_seed(RANDOM_SEED)

np.random.seed(RANDOM_SEED)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Training device:", device)


# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading healthy telemetry...")

df = pd.read_csv(
    TRAIN_PATH
)

values = df[
    FEATURE_COLUMNS
].values.astype(
    np.float32
)

print(
    "Telemetry records:",
    len(values)
)


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


X = create_sequences(
    values,
    SEQUENCE_LENGTH
)


print(
    "Sequence shape:",
    X.shape
)


# ==========================================================
# PYTORCH DATASET
# ==========================================================

X_tensor = torch.tensor(X)


dataset = TensorDataset(
    X_tensor,
    X_tensor
)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================================
# LSTM AUTOENCODER
# ==========================================================

class LSTMAutoencoder(
    nn.Module
):

    def __init__(
        self,
        input_size=7,
        hidden_size=64,
        latent_size=16
    ):

        super().__init__()


        # -------------------------------
        # Encoder
        # -------------------------------

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )


        # -------------------------------
        # Latent projection
        # -------------------------------

        self.latent = nn.Linear(
            hidden_size,
            latent_size
        )


        # -------------------------------
        # Decoder input
        # -------------------------------

        self.decoder_input = nn.Linear(
            latent_size,
            hidden_size
        )


        # -------------------------------
        # Decoder
        # -------------------------------

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )


        # -------------------------------
        # Output layer
        # -------------------------------

        self.output_layer = nn.Linear(
            hidden_size,
            input_size
        )


    def forward(self, x):

        # Encoder
        encoded_sequence, _ = (
            self.encoder(x)
        )

        # Take final time step
        final_hidden = (
            encoded_sequence[:, -1, :]
        )

        # Latent representation
        latent = self.latent(
            final_hidden
        )

        # Expand latent vector
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

        # Decoder
        decoded, _ = self.decoder(
            decoder_input
        )

        # Reconstruct
        output = self.output_layer(
            decoded
        )

        return output


# ==========================================================
# CREATE MODEL
# ==========================================================

model = LSTMAutoencoder()

model = model.to(device)


# ==========================================================
# LOSS
# ==========================================================

criterion = nn.MSELoss()


# ==========================================================
# OPTIMIZER
# ==========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================================
# TRAIN
# ==========================================================

print("\n==============================================")
print("        TRAINING LSTM AUTOENCODER")
print("==============================================")


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_x, _ in dataloader:

        batch_x = batch_x.to(device)

        reconstructed = model(
            batch_x
        )

        loss = criterion(
            reconstructed,
            batch_x
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * len(batch_x)
        )


    epoch_loss = (
        total_loss
        / len(dataset)
    )


    if (
        (epoch + 1) % 5 == 0
        or epoch == 0
    ):

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Loss: {epoch_loss:.6f}"
        )


# ==========================================================
# SAVE
# ==========================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\n==============================================")
print("     LSTM AUTOENCODER TRAINING COMPLETE")
print("==============================================")

print(
    "Model saved to:",
    MODEL_PATH
)