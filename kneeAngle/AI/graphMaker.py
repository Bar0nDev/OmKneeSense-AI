import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv(".\kneeAngle\updated\output\outputCSV\knee_angle_tracked_20260227_192158.csv")

true_angles = df["angle_deg"].values
frames = df["frame"].values

# Assume you already generated predictions
pred_angles = np.array(predicted_angles)  # must match length

plt.figure(figsize=(12,6))

plt.plot(frames, true_angles, label="Ground Truth")
plt.plot(frames, pred_angles, label="Predicted")

plt.xlabel("Frame")
plt.ylabel("Knee Angle (°)")
plt.title("Ground Truth vs Predicted Knee Angle (SEQ=9)")
plt.legend()
plt.grid(True)

plt.show()