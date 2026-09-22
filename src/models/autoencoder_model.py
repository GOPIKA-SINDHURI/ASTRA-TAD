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
    f"{MODEL_DIR}/telemetry_autoencoder.pth"
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

BATCH_SIZE = 64
EPOCHS = 50
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
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Training device:", device)


# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading healthy training data...")

df = pd.read_csv(
    TRAIN_PATH
)

X = df[FEATURE_COLUMNS].values.astype(
    np.float32
)

print("Training samples:", len(X))
print("Features:", X.shape[1])


# ==========================================================
# CONVERT TO PYTORCH TENSOR
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
# AUTOENCODER ARCHITECTURE
# ==========================================================

class TelemetryAutoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(

            nn.Linear(7, 16),

            nn.ReLU(),

            nn.Linear(16, 8),

            nn.ReLU(),

            nn.Linear(8, 3)
        )

        # Decoder
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
# CREATE MODEL
# ==========================================================

model = TelemetryAutoencoder()

model = model.to(device)


# ==========================================================
# LOSS FUNCTION
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
# TRAINING
# ==========================================================

print("\n==============================================")
print("        TRAINING AUTOENCODER")
print("==============================================")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0

    for batch_x, _ in dataloader:

        batch_x = batch_x.to(device)

        # Forward pass
        reconstructed = model(batch_x)

        # Reconstruction loss
        loss = criterion(
            reconstructed,
            batch_x
        )

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += (
            loss.item() *
            len(batch_x)
        )

    epoch_loss = (
        total_loss / len(dataset)
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
# SAVE MODEL
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
print("       AUTOENCODER TRAINING COMPLETE")
print("==============================================")

print("Model saved to:")
print(MODEL_PATH)