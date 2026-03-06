import os
import glob
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# =====================
# CONFIG
# =====================
import torch
import torchvision
print("CUDA available:", torch.cuda.is_available())

print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
VTS_ROOT = "./kneeAngle/updated/output/vts"
CSV_ROOT = "./kneeAngle/updated/output/outputCSV"

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
SEQUENCE_LENGTH = 9   # 🔥 upgraded
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# DATASET
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
                continue

            df = pd.read_csv(csv_path)

            angle_map = {
                int(row["frame"]): row["angle_deg"] / 180.0
                for _, row in df.iterrows()
            }

            frame_paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))

            valid_frames = []
            for path in frame_paths:
                filename = os.path.basename(path)
                frame_idx = int(filename.split("_")[-1].split(".")[0])
                if frame_idx in angle_map:
                    valid_frames.append((path, angle_map[frame_idx]))

            print(f"{session_name}: matched {len(valid_frames)} frames")

            for i in range(half, len(valid_frames) - half):
                window = valid_frames[i-half:i+half+1]
                center_angle = window[half][1]
                frame_paths_window = [w[0] for w in window]
                self.samples.append((frame_paths_window, center_angle))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, angle = self.samples[idx]

        imgs = []
        for path in frame_paths:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        stacked = torch.cat(imgs, dim=0)

        return stacked, torch.tensor(angle, dtype=torch.float32)

# =====================
# MODEL
# =====================

class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

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
                self.backbone.conv1.weight[:, 3*i:3*(i+1), :, :] = old_conv.weight / SEQUENCE_LENGTH

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)

    def forward(self, x):
        return self.backbone(x).squeeze(1)

# =====================
# TRAINING FUNCTIONS
# =====================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for imgs, angles in tqdm(loader):
        imgs = imgs.to(DEVICE)
        angles = angles.to(DEVICE)

        preds = model(imgs)
        loss = criterion(preds, angles)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for imgs, angles in loader:
            imgs = imgs.to(DEVICE)
            angles = angles.to(DEVICE)
            preds = model(imgs)

            preds_all.append(preds.cpu().numpy())
            targets_all.append(angles.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)

    rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2))
    ss_res = np.sum((targets_all - preds_all) ** 2)
    ss_tot = np.sum((targets_all - np.mean(targets_all)) ** 2)
    r2 = 1 - ss_res / ss_tot

    return rmse, r2

# =====================
# MAIN
# =====================

def main():

    sessions = sorted(os.listdir(VTS_ROOT))
    sessions = [s for s in sessions if s.startswith("vts_")]

    train_sessions = sessions[:-2]
    test_sessions = sessions[-2:]

    print("Train sessions:", train_sessions)
    print("Test sessions:", test_sessions)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = VTSDataset(train_sessions, transform)
    test_dataset = VTSDataset(test_sessions, transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = CNNRegressor().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    criterion = nn.SmoothL1Loss()  # 🔥 Huber loss

    best_rmse = float("inf")

    for epoch in range(EPOCHS):

        print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        rmse, r2 = evaluate(model, test_loader)

        scheduler.step(rmse)

        print(
            f"Train Loss: {train_loss:.6f} | "
            f"Test RMSE: {rmse*180:.2f}° | "
            f"Test R²: {r2:.4f}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            torch.save(model.state_dict(), "vts_angle_model_best.pth")
            print("🔥 Saved new BEST model")

    print("Training complete.")

if __name__ == "__main__":
    main()