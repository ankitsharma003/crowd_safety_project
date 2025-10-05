# flow_estimation.py
# Estimate crowd flow direction and suggest movement instructions

import os
from PIL import Image
import numpy as np

def load_images(folder, size=(64, 64), max_frames=20):
    images = []
    for filename in sorted(os.listdir(folder))[:max_frames]:
        if filename.endswith('.tif'):
            img = Image.open(os.path.join(folder, filename)).convert('L')
            img = img.resize(size)
            img_data = np.array(img, dtype=np.float32) / 255.0
            images.append(img_data)
    return images

def estimate_flow(images):
    # Simple frame differencing to estimate movement
    flows = []
    for i in range(1, len(images)):
        flow = images[i] - images[i-1]
        flows.append(flow)
    return flows

def estimate_global_shift(prev: np.ndarray, curr: np.ndarray) -> tuple:
    """
    Estimate global translational shift (dx, dy) between two frames using phase correlation.
    Returns dx in +x rightwards, dy in +y downwards, in pixel units on the 64x64 grid.
    """
    # Convert to float
    a = prev.astype(np.float32)
    b = curr.astype(np.float32)
    # Remove mean to reduce DC component
    a = a - np.mean(a)
    b = b - np.mean(b)
    # FFTs
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    denom = np.abs(R) + 1e-8
    R /= denom
    r = np.fft.ifft2(R)
    r = np.abs(r)
    max_idx = np.unravel_index(np.argmax(r), r.shape)
    peak_y, peak_x = int(max_idx[0]), int(max_idx[1])
    h, w = a.shape
    # Convert to signed shift
    dy = peak_y if peak_y < h/2 else peak_y - h
    dx = peak_x if peak_x < w/2 else peak_x - w
    return float(dx), float(dy)

def estimate_shift_and_confidence(prev: np.ndarray, curr: np.ndarray) -> tuple:
    """
    Returns (dx, dy, confidence) where confidence is in [0,1].
    Confidence is computed as (peak - second_peak) / (peak + 1e-8).
    """
    a = prev.astype(np.float32)
    b = curr.astype(np.float32)
    a = a - np.mean(a)
    b = b - np.mean(b)
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    R = Fa * np.conj(Fb)
    denom = np.abs(R) + 1e-8
    R /= denom
    r = np.fft.ifft2(R)
    r = np.abs(r)
    flat = r.ravel()
    max_idx = int(np.argmax(flat))
    peak = float(flat[max_idx])
    # zero out peak and find second
    flat_no_peak = flat.copy()
    flat_no_peak[max_idx] = 0.0
    second = float(np.max(flat_no_peak))
    confidence = float(max(0.0, min(1.0, (peak - second) / (peak + 1e-8))))
    peak_y, peak_x = np.unravel_index(max_idx, r.shape)
    h, w = a.shape
    dy = peak_y if peak_y < h/2 else peak_y - h
    dx = peak_x if peak_x < w/2 else peak_x - w
    return float(dx), float(dy), confidence

def estimate_shifts_confidence(images, alpha: float = 0.6):
    """
    Estimate per-pair shifts with confidence and return smoothed shifts.
    Returns list of dicts: {'dx': dx_smooth, 'dy': dy_smooth, 'confidence': conf}
    """
    results = []
    dx_s, dy_s = 0.0, 0.0
    initialized = False
    for i in range(1, len(images)):
        dx, dy, conf = estimate_shift_and_confidence(images[i-1], images[i])
        if not initialized:
            dx_s, dy_s = dx, dy
            initialized = True
        else:
            dx_s = alpha * dx + (1 - alpha) * dx_s
            dy_s = alpha * dy + (1 - alpha) * dy_s
        results.append({'dx': dx_s, 'dy': dy_s, 'confidence': conf})
    return results

def dominant_direction_from_pair(prev: np.ndarray, curr: np.ndarray) -> str:
    dx, dy = estimate_global_shift(prev, curr)
    if abs(dx) < 1e-3 and abs(dy) < 1e-3:
        return "No movement detected"
    if abs(dx) > abs(dy):
        return "Suggest moving right (East)" if dx > 0 else "Suggest moving left (West)"
    else:
        return "Suggest moving down (South)" if dy > 0 else "Suggest moving up (North)"

def dominant_direction(flow):
    # Compute average movement in x and y
    h, w = flow.shape
    y_indices, x_indices = np.mgrid[0:h, 0:w]
    total = np.sum(flow)
    if total == 0:
        return "No movement detected"
    x_flow = np.sum(flow * x_indices) / total - w/2
    y_flow = np.sum(flow * y_indices) / total - h/2
    if abs(x_flow) > abs(y_flow):
        if x_flow > 0:
            return "Suggest moving right (East)"
        else:
            return "Suggest moving left (West)"
    else:
        if y_flow > 0:
            return "Suggest moving down (South)"
        else:
            return "Suggest moving up (North)"

if __name__ == "__main__":
    folder = r"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
    images = load_images(folder)
    flows = estimate_flow(images)
    # Analyze last flow
    if flows:
        instruction = dominant_direction(flows[-1])
        print("Crowd flow instruction:", instruction)
    else:
        print("Not enough frames to estimate flow.")
