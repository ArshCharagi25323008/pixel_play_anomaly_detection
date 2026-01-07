# ============================================================
# LEADERBOARD-MAX TEMPORAL AUTOENCODER (AP-OPTIMIZED)
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# ======================
# CONFIG
# ======================
# Root dataset directory
DATASET_ROOT = "Dataset"

# Path to training videos (used ONLY for training)
TRAIN_PATH = os.path.join(DATASET_ROOT, "training_videos")

# Path to cleaned testing videos (NOT used in this script, but kept for consistency)
TEST_PATH  = os.path.join(DATASET_ROOT, "cleaned_testing_videos")

# Image resolution for model input
IMG_SIZE = 128

# Number of consecutive frames per temporal cuboid
TEMPORAL_LENGTH = 16

# Sliding window stride for cuboid generation
STRIDE = 2

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3

# Device selection: CUDA → MPS → CPU
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Device:", DEVICE)

# ======================
# DATASET
# ======================
class TemporalDataset(Dataset):
    """
    Dataset that converts videos into temporal cuboids of frames.

    Each sample is a stack of TEMPORAL_LENGTH grayscale frames,
    concatenated along the channel dimension.
    """
    def __init__(self, root, train=True):
        self.samples = []
        self.train = train

        # Resize and normalize frames to tensors
        self.transform = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor()
        ])

        # Iterate over all video folders
        for vid in sorted(os.listdir(root)):
            vpath = os.path.join(root, vid)
            if not os.path.isdir(vpath):
                continue

            # Collect all frames for the video
            frames = sorted([f for f in os.listdir(vpath) if f.endswith(".jpg")])
            max_start = len(frames) - TEMPORAL_LENGTH

            # Generate temporal windows using a sliding window
            for i in range(0, max_start + 1, STRIDE):
                self.samples.append((vid, frames, i))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vid, frames, start = self.samples[idx]
        cuboid = []

        # Load TEMPORAL_LENGTH consecutive frames
        for i in range(TEMPORAL_LENGTH):
            img = Image.open(
                os.path.join(TRAIN_PATH if self.train else TEST_PATH,
                             vid, frames[start+i])
            ).convert("L")

            # Apply resize + tensor transform
            x = self.transform(img)

            # ---- AGGRESSIVE AUGMENTATION (TRAIN ONLY)
            # These augmentations intentionally corrupt training data
            # so that reconstruction error becomes sensitive to anomalies
            if self.train:
                # Random additive noise
                if np.random.rand() < 0.3:
                    x += torch.empty_like(x).uniform_(-0.15, 0.15)
                    x = torch.clamp(x, 0, 1)

                # Random vertical flip
                if np.random.rand() < 0.2:
                    x = torch.flip(x, dims=[1])

            cuboid.append(x)

        # Concatenate frames along channel dimension
        # Shape: (TEMPORAL_LENGTH, H, W)
        return torch.cat(cuboid, dim=0)

# ======================
# MODEL
# ======================
class TemporalAE(nn.Module):
    """
    Temporal convolutional autoencoder.

    The model reconstructs stacked temporal cuboids.
    High reconstruction error indicates abnormal frames.
    """
    def __init__(self):
        super().__init__()

        # Encoder compresses temporal information
        self.encoder = nn.Sequential(
            nn.Conv2d(TEMPORAL_LENGTH, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),
        )

        # Decoder reconstructs the original cuboid
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, TEMPORAL_LENGTH, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# ======================
# TRAIN
# ======================
# Create training dataset and loader
train_ds = TemporalDataset(TRAIN_PATH, train=True)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model, optimizer, and reconstruction loss
model = TemporalAE().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

print(f"Training on {len(train_ds)} cuboids")

# Training loop
for ep in range(EPOCHS):
    model.train()
    losses = []

    for x in tqdm(train_dl, desc=f"Epoch {ep+1}/{EPOCHS}"):
        x = x.to(DEVICE)

        # Forward pass
        recon = model(x)

        # Reconstruction loss
        loss = loss_fn(recon, x)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    print(f"Epoch {ep+1} Loss: {np.mean(losses):.6f}")

# Save trained autoencoder weights
torch.save(model.state_dict(), "temporal_ae.pth")
