# Pixel Play – Video Anomaly Detection

This repository contains my solutions for the **Pixel Play video anomaly detection challenge**.  
The task is to assign an anomaly score to **every frame** of a video sequence using only **unlabeled training data**.

I intentionally include **two fundamentally different approaches**:

1. **Leaderboard / AP-optimized solution** – tuned specifically to maximize Average Precision (AP)
2. **Real-world motion-based solution** – designed to detect genuine motion anomalies

This distinction is intentional and explained clearly below.

---

## Repository Structure
```text
.
├── flip_correction.py # Detects & fixes vertically flipped test frames
├── train_temporal_ae.py # Trains temporal autoencoder (AP-optimized)
├── infer_temporal_ae.py # Inference + submission generation
├── raft_motion_model.py # Real-world motion-based anomaly detection
├── submission.csv
└── README.md
```

---

## Dataset Assumptions

- Videos are provided as folders of `.jpg` frames
- Training videos contain **only normal behavior**
- Test videos may contain:
  - Motion anomalies
  - Visual distortions
  - Corrupted frames
  - Vertically flipped frames
- Evaluation metric: **Average Precision (AP)**

---

## Best Performing Approach (Leaderboard / AP-Optimized)

### Overview

This solution is **explicitly optimized for leaderboard performance**, not real-world anomaly semantics.

The core idea is simple:

> Frames that are hardest to reconstruct *relative to the global test distribution* should receive the highest anomaly scores.

Since AP depends only on **ranking**, this approach focuses on producing a strong ordering rather than accurate semantic detection.

---

### Pipeline

#### 1. Flip Correction (Dataset Cleaning)

- A lightweight CNN is trained using training frames
- The network learns to distinguish **normal vs vertically flipped frames**
- Test frames predicted as flipped are corrected
- Cleaned frames are saved to `cleaned_testing_videos/`

This step removes dataset-specific corruption that would otherwise confuse the anomaly model.

---

#### 2. Temporal Autoencoder Training

- A convolutional autoencoder is trained on **temporal cuboids** (multiple consecutive frames)
- Frames are:
  - Converted to grayscale
  - Resized
  - Concatenated along the channel dimension
- Aggressive augmentation is applied:
  - Random noise injection
  - Random vertical flips

The model learns dominant visual patterns rather than precise motion dynamics.

---

#### 3. Inference & AP-Oriented Post-Processing

- Reconstruction error is computed per frame
- Errors from overlapping temporal windows are aggregated
- Scores are:
  - Globally normalized across the entire test set
  - Lightly temporally smoothed
  - Converted to global ranks

Final predictions are written to `submission.csv`.

---

### Why This Scores High on AP

- Produces a **heavy-tailed score distribution**
- Strongly separates a small number of extreme frames
- Exploits dataset-specific visual artifacts
- Optimizes ranking quality rather than correctness

**Important:**  
This approach is *not suitable for real-world anomaly detection*.

---

## Real-World Motion-Based Solution

### Overview

This solution focuses on **motion consistency**, not visual reconstruction.

Core idea:

> Anomalies correspond to **unexpected motion patterns**, not corrupted appearance.

---

### Pipeline

- Dense optical flow is computed using **RAFT-Large**
- Flow fields are compressed and vectorized
- A GRU predicts future motion from past motion
- Anomaly score = motion prediction error
- Scores are temporally smoothed and normalized

---

### Why This Is More Realistic

- Explicitly models temporal dynamics
- Detects unusual motion behavior
- Robust to visual noise and distortions
- Better generalization to real-world scenarios

---

### Why It Scores Lower on AP

- Motion anomalies are rarer than visual artifacts
- Errors are smoother and less extreme
- AP favors sharp outliers over semantic correctness

This clearly demonstrates the difference between **metric optimization** and **problem correctness**.

---

## Key Takeaway

Optimizing for a leaderboard metric is not the same as solving the real problem.

- The **AP-optimized solution** demonstrates how ranking-based metrics can be maximized
- The **motion-based solution** demonstrates principled anomaly detection

Both approaches are included intentionally.

---

## Requirements

- Python 3.9+
- PyTorch
- torchvision
- OpenCV
- NumPy
- tqdm
- PIL

---

## Usage

1. (Optional) Clean flipped frames  
   Run `flip_correction.py`

2. Train the temporal autoencoder  
   Run `train_temporal_ae.py`

3. Run inference and generate submission  
   Run `infer_temporal_ae.py`

4. Run the real-world motion-based model  
   Run `raft_motion_model.py`

---

## Author

**Arsh Charagi**

This project explores the trade-offs between leaderboard optimization and real-world anomaly detection.
# pixel_play_anomaly_detection
# pixel_play_anomaly_detection
# pixel_play_anomaly_detection
