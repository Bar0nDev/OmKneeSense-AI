import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# =====================
# CONFIG (MUST MATCH TRAINING)
# =====================

SEQUENCE_LENGTH = 9
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VTS_ROOT = "./kneeAngle/updated/output/vts"
CSV_ROOT = "./kneeAngle/updated/output/outputCSV"

MODEL_PATH = "./kneeAngle/vts_angle_model_best_SEQ9.pth"
OUTPUT_DIR = "./kneeAngle/eval_results_seq9"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("CUDA available:", torch.cuda.is_available())

# =====================
# MODEL (IDENTICAL TO TRAINING)
# =====================

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )

        old_conv = self.backbone.conv1

        self.backbone.conv1 = nn.Conv2d(
            in_channels=3 * SEQUENCE_LENGTH,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )

        with torch.no_grad():
            for i in range(SEQUENCE_LENGTH):
                self.backbone.conv1.weight[:, 3*i:3*(i+1), :, :] = (
                    old_conv.weight / SEQUENCE_LENGTH
                )

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features, 1
         )

    def forward(self, x):
        return self.backbone(x).squeeze(1)

# =====================
# DATASET (UPDATED: RETURNS FRAME + SESSION)
# =====================

class VTSDataset(Dataset):
    def __init__(self, sessions, transform=None):
        self.samples = []
        self.transform = transform
        half = SEQUENCE_LENGTH // 2

        for session_name in sessions:

            frames_dir = os.path.join(VTS_ROOT, session_name, "frames")
            timestamp = session_name.replace("vts_", "")
            csv_name = f"knee_angle_tracked_{timestamp}.csv"
            csv_path = os.path.join(CSV_ROOT, csv_name)

            if not os.path.exists(csv_path):
                print(f"CSV missing for {session_name}")
                continue

            df = pd.read_csv(csv_path)

            angle_map = {
                int(row["frame"]): row["angle_deg"] / 180.0
                for _, row in df.iterrows()
            }

            frame_paths = sorted(
                glob.glob(os.path.join(frames_dir, "*.png"))
            )

            valid_frames = []
            for path in frame_paths:
                filename = os.path.basename(path)
                frame_idx = int(filename.split("_")[-1].split(".")[0])
                if frame_idx in angle_map:
                    valid_frames.append((path, frame_idx, angle_map[frame_idx]))

            print(f"{session_name}: matched {len(valid_frames)} frames")

            for i in range(half, len(valid_frames) - half):
                window = valid_frames[i-half:i+half+1]
                center = window[half]

                frame_paths_window = [w[0] for w in window]
                center_frame_idx = center[1]
                center_angle = center[2]

                self.samples.append(
                    (frame_paths_window, center_angle, center_frame_idx, session_name)
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, angle, frame_idx, session = self.samples[idx]

        imgs = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        stacked = torch.cat(imgs, dim=0)

        return stacked, torch.tensor(angle, dtype=torch.float32), frame_idx, session

# =====================
# TRANSFORM
# =====================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# LOAD MODEL
# =====================

model = CNNRegressor().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("Model loaded successfully.")

# =====================
# SELECT TEST SESSIONS
# =====================

sessions = sorted(os.listdir(VTS_ROOT))
sessions = [s for s in sessions if s.startswith("vts_")]

test_sessions = sessions[-2:]  # change if needed
print("Test sessions:", test_sessions)

test_dataset = VTSDataset(test_sessions, transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# =====================
# EVALUATION
# =====================

all_results = []

with torch.no_grad():
    for imgs, angles, frame_idxs, sessions_batch in tqdm(test_loader):
        imgs = imgs.to(DEVICE)
        angles = angles.to(DEVICE)

        preds = model(imgs)

        preds_deg = preds.cpu().numpy() * 180.0
        targets_deg = angles.cpu().numpy() * 180.0

        for i in range(len(preds_deg)):
            all_results.append({
                "session": sessions_batch[i],
                "frame": int(frame_idxs[i]),
                "true_angle": targets_deg[i],
                "pred_angle": preds_deg[i],
                "abs_error": abs(preds_deg[i] - targets_deg[i])
            })

results_df = pd.DataFrame(all_results)

# =====================
# GLOBAL METRICS
# =====================

rmse = np.sqrt(np.mean((results_df["pred_angle"] - results_df["true_angle"]) ** 2))
mae = np.mean(results_df["abs_error"])
bias = np.mean(results_df["pred_angle"] - results_df["true_angle"])

within_5 = np.mean(results_df["abs_error"] <= 5.0)
within_10 = np.mean(results_df["abs_error"] <= 10.0)
within_15 = np.mean(results_df["abs_error"] <= 15.0)

print("\n===== FINAL TEST RESULTS =====")
print(f"RMSE: {rmse:.2f}°")
print(f"MAE: {mae:.2f}°")
print(f"Bias: {bias:.2f}°")
print(f"±5° Accuracy:  {within_5*100:.2f}%")
print(f"±10° Accuracy: {within_10*100:.2f}%")
print(f"±15° Accuracy: {within_15*100:.2f}%")

# Save global results
results_df.to_csv(os.path.join(OUTPUT_DIR, "all_predictions_seq9.csv"), index=False)

# =====================
# PER-SESSION SAVE + PLOT
# =====================

for session in results_df["session"].unique():

    session_df = results_df[results_df["session"] == session]
    session_df = session_df.sort_values("frame")

    session_df.to_csv(
        os.path.join(OUTPUT_DIR, f"{session}_predictions.csv"),
        index=False
    )

    plt.figure(figsize=(12,6))
    plt.plot(session_df["frame"], session_df["true_angle"], label="True")
    plt.plot(session_df["frame"], session_df["pred_angle"], label="Predicted")
    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title(f"True vs Predicted Angle - {session}")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(OUTPUT_DIR, f"{session}_plot.png"))
    plt.close()

print("\nAll predictions and plots saved to:", OUTPUT_DIR)