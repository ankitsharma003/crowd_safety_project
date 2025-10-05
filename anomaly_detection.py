# Step 5: Anomaly Detection
# This script flags frames with high reconstruction error as anomalies.

import pickle
import math

if __name__ == "__main__":
    with open("pca_results.pkl", "rb") as f:
        results = pickle.load(f)
    errors = results['errors']
    # Set threshold: mean + 2*std
    mean_err = sum(errors) / len(errors)
    std_err = math.sqrt(sum((e - mean_err) ** 2 for e in errors) / len(errors))
    threshold = mean_err + 2 * std_err
    anomalies = [i for i, e in enumerate(errors) if e > threshold]
    print(f"Threshold: {threshold:.4f}")
    print(f"Anomalous frames: {anomalies}")
    # Save anomaly indices
    with open("anomalies.pkl", "wb") as f:
        pickle.dump(anomalies, f)
    print("Anomaly detection complete. Indices saved.")
