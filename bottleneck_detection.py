# bottleneck_detection.py
# Detect bottlenecks in crowd flow using density and movement analysis

import numpy as np
from flow_estimation import load_images, estimate_flow

def detect_bottlenecks(images, flows, density_threshold=0.55, movement_threshold=0.04):
    bottlenecks = []
    for i, img in enumerate(images):
        density = np.mean(img)
        if i == 0:
            movement = 0
        else:
            movement = np.mean(np.abs(flows[i-1]))
        # Bottleneck: high density and low movement
        if density > density_threshold and movement < movement_threshold:
            bottlenecks.append(i)
    return bottlenecks

if __name__ == "__main__":
    folder = r"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
    images = load_images(folder)
    flows = estimate_flow(images)
    bottlenecks = detect_bottlenecks(images, flows)
    print(f"Bottleneck frames: {bottlenecks}")
    if bottlenecks:
        print("Warning: Potential crowd congestion detected in these frames.")
    else:
        print("No bottlenecks detected.")
