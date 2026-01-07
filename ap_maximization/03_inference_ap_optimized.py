# ============================================================
# LEADERBOARD-MAX TEMPORAL AUTOENCODER (AP-OPTIMIZED, INFERENCE)
# ============================================================

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d

# ======================
# CONFIG
# ======================
# Root dataset directory
DATASET_ROOT = "Dataset"

# Training path kept only for consistency (not used here)
TRAIN_PATH = os.path.join(DATASET_ROOT, "training_videos")

# Path to cleaned testing videos (after flip correction)
TEST_PATH  = os.path.join(DATASET_ROOT, "cleaned_testing_videos")

# Input resolution expected by the trained model
IMG_SIZE = 128

# Number of frames per temporal cuboid
TEMPORAL_LENGTH = 16

# Sliding window stride
# STRIDE = 1 gives dense overlap → better temporal localization
STRIDE = 1

# Temporal smoothing window size
# Small value preserves sharp anomalies while reducing noise
SMOOTH = 3

# Path to pretrained autoencoder weights
MODEL_PATH = "temporal_ae.pth"

# Device selection
DEVICE = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Device:", DEVICE)

# ======================
# MODEL
# ======================
class TemporalAE(nn.Module):
    """
    Temporal convolutional autoencoder used for anomaly detection.
    Reconstructs stacked temporal cuboids.
    """
    def __init__(self):
        super().__init__()

        # Encoder compresses spatio-temporal information
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
# LOAD TRAINED MODEL (NO RETRAIN)
# ======================
# Initialize model architecture
model = TemporalAE().to(DEVICE)

# Load pretrained weights from training script
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

# Set model to evaluation mode
model.eval()
print("Loaded trained model:", MODEL_PATH)

# ======================
# TEST / ERROR COMPUTATION
# ======================
# Dictionary to store per-video frame-wise reconstruction errors
errors = {}

# Define transforms (kept separate for clarity)
resize = T.Resize((IMG_SIZE, IMG_SIZE))
to_tensor = T.ToTensor()

print("Scoring test videos...")

# Iterate over each test video
for vid in sorted(os.listdir(TEST_PATH)):
    vpath = os.path.join(TEST_PATH, vid)
    if not os.path.isdir(vpath):
        continue

    # Collect all frames of the video
    frames = sorted([f for f in os.listdir(vpath) if f.endswith(".jpg")])

    # Each frame will accumulate multiple error values
    frame_errs = [[] for _ in frames]

    # Slide over video using temporal cuboids
    for i in range(0, len(frames) - TEMPORAL_LENGTH + 1, STRIDE):
        cuboid = []

        # Load TEMPORAL_LENGTH consecutive frames
        for j in range(TEMPORAL_LENGTH):
            img = Image.open(os.path.join(vpath, frames[i+j])).convert("L")
            cuboid.append(resize(to_tensor(img)))

        # Stack frames along channel dimension
        x = torch.cat(cuboid, dim=0).unsqueeze(0).to(DEVICE)

        # Forward pass (no gradients needed)
        with torch.no_grad():
            recon = model(x)

        # Compute per-frame reconstruction error
        per_frame = ((x - recon) ** 2).mean(dim=[2, 3]).squeeze().cpu().numpy()

        # Assign each error to the corresponding frame
        for j in range(TEMPORAL_LENGTH):
            frame_errs[i + j].append(per_frame[j])

    # Average accumulated errors per frame
    errors[vid] = np.array([np.mean(e) if e else 0.0 for e in frame_errs])

# ======================
# GLOBAL NORMALIZATION + SMOOTHING
# ======================
# Flatten all errors across all videos
all_scores = np.concatenate(list(errors.values()))

# Global min and high-percentile max for robust normalization
mn = all_scores.min()
mx = np.percentile(all_scores, 99.99)

submission = []

# Process each video independently
for vid in sorted(errors):
    raw = errors[vid]

    # Normalize scores globally into [0,1]
    norm = np.clip((raw - mn) / (mx - mn + 1e-8), 0, 1)

    # Temporal smoothing AFTER normalization
    smooth = uniform_filter1d(norm, size=SMOOTH)

    # Map smoothed scores to submission format
    frames = sorted(os.listdir(os.path.join(TEST_PATH, vid)))
    for idx, s in enumerate(smooth):
        frame_num = int(frames[idx].split("_")[1].split(".")[0])
        submission.append((f"{int(vid)}_{frame_num}", round(float(s), 6)))

# ======================
# SAVE SUBMISSION
# ======================
# Create submission DataFrame
df = pd.DataFrame(submission, columns=["Id", "Predicted"])

# Ensure strict ordering by video and frame index
df[["v","f"]] = df["Id"].str.split("_", expand=True).astype(int)
df = df.sort_values(["v","f"]).drop(columns=["v","f"])

# ======================
# GLOBAL RANK NORMALIZATION
# ======================
# Convert scores to ranks to optimize AP
from scipy.stats import rankdata

scores = df["Predicted"].values
df["Predicted"] = rankdata(scores) / len(scores)

# Write final CSV
df.to_csv("submission.csv", index=False)
print("submission.csv written:", len(df))
print(df.head())
