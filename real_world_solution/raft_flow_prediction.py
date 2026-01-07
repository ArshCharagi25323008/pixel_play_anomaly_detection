import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from tqdm import tqdm
import csv

# =====================
# DEVICE
# =====================
# Use Apple MPS (Mac GPU) if available, otherwise CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# =====================
# CONFIG
# =====================
# Dataset paths
DATASET_ROOT = "Dataset"
TRAIN_DIR = os.path.join(DATASET_ROOT, "training_videos")
TEST_DIR = os.path.join(DATASET_ROOT, "testing_videos")

# Temporal prediction settings
SEQ_LEN = 8        # Number of past optical flow frames used for prediction
EPOCHS = 8
BATCH = 64
LR = 1e-3
SMOOTH = 7         # Temporal smoothing window
FLOW_SIZE = (64, 64)   # Spatial compression of optical flow

# =====================
# LOAD RAFT LARGE
# =====================
# RAFT is a state-of-the-art optical flow model
# We use it as a frozen feature extractor for motion
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

raft = raft_large(weights=Raft_Large_Weights.DEFAULT).to(device)
raft.eval()  # inference-only

# Resize optical flow to reduce dimensionality
resize_flow = Resize(FLOW_SIZE)

@torch.no_grad()
def compute_flow(img1, img2):
    """
    Computes optical flow between two RGB frames using RAFT.
    Output is spatially downsampled to FLOW_SIZE.
    """
    img1 = torch.from_numpy(img1).permute(2,0,1).float() / 255
    img2 = torch.from_numpy(img2).permute(2,0,1).float() / 255
    img1 = img1.unsqueeze(0).to(device)
    img2 = img2.unsqueeze(0).to(device)

    # RAFT returns a list of flow estimates; we take the final one
    flow = raft(img1, img2)[-1]
    flow = resize_flow(flow)

    return flow.squeeze().cpu().numpy()

def flow_to_vec(flow):
    """
    Flattens the optical flow field into a 1D vector.
    """
    return flow.reshape(-1)

# =====================
# FLOW PREDICTOR
# =====================
class FlowGRU(nn.Module):
    """
    GRU-based predictor that learns temporal dynamics of optical flow.
    Given SEQ_LEN past flow vectors, it predicts the next flow.
    """
    def __init__(self, dim):
        super().__init__()
        self.gru = nn.GRU(dim, 512, batch_first=True)
        self.fc = nn.Linear(512, dim)

    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))

# =====================
# LOAD TRAIN FLOWS
# =====================
print("Extracting TRAIN flows...")
train_X, train_Y = [], []

# Iterate over training videos
for vid in sorted(os.listdir(TRAIN_DIR)):
    vpath = os.path.join(TRAIN_DIR, vid)
    if not os.path.isdir(vpath):
        continue

    # Load all frames for the video
    frames = sorted([f for f in os.listdir(vpath) if f.endswith(".jpg")])
    imgs = []

    for f in frames:
        img = cv2.imread(os.path.join(vpath, f))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Compute optical flow between consecutive frames
    flows = []
    for i in range(len(imgs)-1):
        flows.append(flow_to_vec(compute_flow(imgs[i], imgs[i+1])))

    flows = np.array(flows)

    # Create sliding-window sequences for GRU training
    for i in range(len(flows)-SEQ_LEN):
        train_X.append(flows[i:i+SEQ_LEN])    # past flows
        train_Y.append(flows[i+SEQ_LEN])      # next flow

# Convert to tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
train_Y = torch.tensor(train_Y, dtype=torch.float32)

# Initialize model and optimizer
model = FlowGRU(train_X.shape[-1]).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.MSELoss()

dataset = torch.utils.data.TensorDataset(train_X, train_Y)
loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH, shuffle=True)

# =====================
# TRAIN
# =====================
print("Training flow predictor...")
for ep in range(EPOCHS):
    losses = []
    model.train()

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        pred = model(xb)
        loss = loss_fn(pred, yb)

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())

    print(f"Epoch {ep+1}/{EPOCHS} | Loss {np.mean(losses):.6f}")

# =====================
# TEST
# =====================
print("Scoring TEST videos...")
model.eval()
ids, scores = [], []

# Iterate over test videos
for vid in sorted(os.listdir(TEST_DIR)):
    vpath = os.path.join(TEST_DIR, vid)
    if not os.path.isdir(vpath):
        continue

    frames = sorted([f for f in os.listdir(vpath) if f.endswith(".jpg")])
    imgs = []

    for f in frames:
        img = cv2.imread(os.path.join(vpath, f))
        imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Compute optical flows for the test video
    flows = []
    for i in range(len(imgs) - 1):
        flows.append(flow_to_vec(compute_flow(imgs[i], imgs[i+1])))
    flows = np.array(flows)

    # Initialize per-frame anomaly scores
    frame_scores = np.zeros(len(frames))

    # Predict next flow and measure prediction error
    for i in range(SEQ_LEN, len(flows)):
        x = torch.tensor(flows[i-SEQ_LEN:i]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x).cpu().numpy()

        # Prediction error = anomaly score
        frame_scores[i+1] = np.linalg.norm(pred - flows[i])
        # i+1 because flow[i] corresponds to frame i+1

    # Temporal smoothing for stability
    frame_scores = np.convolve(
        frame_scores,
        np.ones(SMOOTH) / SMOOTH,
        mode="same"
    )

    # Save scores for ALL frames (important for submission format)
    for f, s in zip(frames, frame_scores):
        idx = int(f.split("_")[1].split(".")[0])
        ids.append(f"{int(vid)}_{idx}")
        scores.append(s)

# =====================
# NORMALIZE + SAVE
# =====================
# Global min-max normalization
scores = np.array(scores)
scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

# Write submission CSV
with open("submission.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Id", "Predicted"])
    for i, s in zip(ids, scores):
        w.writerow([i, f"{s:.6f}"])

print("submission.csv written")
