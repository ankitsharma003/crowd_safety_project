# Step 1: Data Preparation
# This script loads .tif images, resizes, normalizes, and optionally applies background subtraction.

import os
from PIL import Image

def load_images(folder, size=(64, 64), background_subtract=False):
    images = []
    prev_img = None
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.tif'):
            path = os.path.join(folder, filename)
            img = Image.open(path).convert('L')  # Grayscale
            img = img.resize(size)
            img_data = [pixel / 255.0 for pixel in img.getdata()]
            if background_subtract and prev_img is not None:
                img_data = [abs(a - b) for a, b in zip(img_data, prev_img)]
            images.append(img_data)
            prev_img = img_data
    return images

if __name__ == "__main__":
    # Example usage: load images from Test001
    folder = r"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
    images = load_images(folder, background_subtract=True)
    print(f"Loaded {len(images)} images. Each image vector length: {len(images[0])}")
