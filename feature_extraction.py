# Step 2: Feature Extraction
# This script loads preprocessed images and stacks them into a spatiotemporal matrix.

import pickle
from data_preparation import load_images

if __name__ == "__main__":
    folder = r"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
    images = load_images(folder, background_subtract=True)
    # images is already a list of 1D vectors
    spatiotemporal_matrix = images  # Each row: frame; Each column: pixel
    print(f"Spatiotemporal matrix shape: {len(spatiotemporal_matrix)} frames x {len(spatiotemporal_matrix[0])} features")
    # Save matrix for later steps
    with open("spatiotemporal_matrix.pkl", "wb") as f:
        pickle.dump(spatiotemporal_matrix, f)
    print("Saved spatiotemporal_matrix.pkl")
