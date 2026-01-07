import os, shutil, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# ================= CONFIG =================
# Paths for training videos, raw testing videos, and cleaned testing output
TRAIN_PATH = "Dataset/training_videos"
TEST_PATH  = "Dataset/testing_videos"
OUT_PATH   = "Dataset/cleaned_testing_videos"

# Image size used for flip detection (small is sufficient)
IMG = 64

# Training hyperparameters for flip detector
BATCH = 64
EPOCHS = 5
LR = 1e-3

# Select device (CUDA if available, otherwise MPS)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("Device:", DEVICE)

# ================= DATA =================
class FlipDataset(Dataset):
    """
    Self-supervised dataset for flip detection.

    Each image is used twice:
    - original image labeled as 0 (normal)
    - vertically flipped image labeled as 1 (flipped)

    This avoids the need for manual labels.
    """
    def __init__(self, root, limit=3000):
        self.samples = []
        frames = []

        # Collect all frame paths from training videos
        for v in os.listdir(root):
            vp = os.path.join(root, v)
            if not os.path.isdir(vp): 
                continue
            for f in os.listdir(vp):
                if f.endswith(".jpg"):
                    frames.append(os.path.join(vp, f))

        # Randomly subsample frames to keep training fast
        frames = random.sample(frames, min(limit, len(frames)))

        # Create paired samples: (normal, flipped)
        for p in frames:
            self.samples.append((p, 0))  # normal image
            self.samples.append((p, 1))  # vertically flipped image

        # Basic preprocessing: resize + tensor conversion
        self.t = T.Compose([T.Resize((IMG,IMG)), T.ToTensor()])

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, i):
        # Load image and label
        p, y = self.samples[i]
        img = Image.open(p).convert("L")

        # Apply preprocessing
        x = self.t(img)

        # If label indicates flipped, apply vertical flip
        if y == 1:
            x = TF.vflip(x)

        return x, torch.tensor(y)

# ================= MODEL =================
class FlipCNN(nn.Module):
    """
    Lightweight CNN for binary classification:
    - class 0: normal frame
    - class 1: vertically flipped frame
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,2,1), nn.ReLU(),
            nn.Conv2d(16,32,3,2,1), nn.ReLU(),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64,2)
        )

    def forward(self,x): 
        return self.net(x)

# ================= TRAIN =================
# Create dataset and dataloader for flip detection
ds = FlipDataset(TRAIN_PATH)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

# Initialize model, optimizer, and loss function
model = FlipCNN().to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

print("Training flip detector...")
for e in range(EPOCHS):
    for x,y in tqdm(dl):
        # Move data to device
        x,y = x.to(DEVICE), y.to(DEVICE)

        # Forward pass and loss computation
        loss = loss_fn(model(x), y)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {e+1}/{EPOCHS}")

# ================= CLEAN TEST =================
print("Cleaning test videos...")

# Create output directory for cleaned test videos
os.makedirs(OUT_PATH, exist_ok=True)

# Switch model to evaluation mode
model.eval()

with torch.no_grad():
    # Iterate through test videos
    for v in tqdm(sorted(os.listdir(TEST_PATH))):
        vp = os.path.join(TEST_PATH, v)
        if not os.path.isdir(vp): 
            continue

        # Create corresponding output folder
        out_v = os.path.join(OUT_PATH, v)
        os.makedirs(out_v, exist_ok=True)

        # Process each frame
        for f in os.listdir(vp):
            if not f.endswith(".jpg"): 
                continue

            p = os.path.join(vp, f)
            img = Image.open(p).convert("L")

            # Prepare image for model inference
            x = T.Resize((IMG,IMG))(img)
            x = T.ToTensor()(x).unsqueeze(0).to(DEVICE)

            # Predict flip probability
            prob = torch.softmax(model(x),1)[0,1].item()

            # If model predicts flipped, correct it
            if prob > 0.5:
                img = TF.vflip(img)

            # Save cleaned frame
            img.save(os.path.join(out_v, f))

print("✅ cleaned_testing_videos created")
