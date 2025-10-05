# Step 6: Visualization
# This script visualizes anomaly maps for flagged frames.

import pickle
import os
from PIL import Image

if __name__ == "__main__":
    # Load data
    with open("spatiotemporal_matrix.pkl", "rb") as f:
        original_frames = pickle.load(f)
    with open("pca_results.pkl", "rb") as f:
        results = pickle.load(f)
    reconstructions = results['reconstructions']
    with open("anomalies.pkl", "rb") as f:
        anomalies = pickle.load(f)
    # Visualization settings
    size = (64, 64)  # Must match preprocessing
    out_dir = "anomaly_maps"
    os.makedirs(out_dir, exist_ok=True)
    # For each anomalous frame, save anomaly map
    for idx in anomalies:
        diff = [abs(original_frames[idx][j] - reconstructions[idx][j]) for j in range(len(original_frames[idx]))]
        # Normalize diff to [0,255]
        max_diff = max(diff)
        img_data = [int(255 * (d / max_diff)) if max_diff > 0 else 0 for d in diff]
        img = Image.new('L', size)
        img.putdata(img_data)
        img.save(os.path.join(out_dir, f"anomaly_{idx:04d}.png"))
    print(f"Saved {len(anomalies)} anomaly maps to {out_dir}/")
