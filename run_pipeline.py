# run_pipeline.py
# Runs the full crowd anomaly detection pipeline step by step.

import os
import sys

# Step 1: Data Preparation
os.system('python data_preparation.py')
# Step 2: Feature Extraction
os.system('python feature_extraction.py')
# Step 3: Autoencoder Training
os.system('python autoencoder.py')
# Step 4: Low-Rank Decomposition
os.system('python low_rank_decomposition.py')
# Step 5: Anomaly Detection
os.system('python anomaly_detection.py')
# Step 6: Visualization
os.system('python visualization.py')

print("\nPipeline complete. Check anomaly_maps/ for results.")
