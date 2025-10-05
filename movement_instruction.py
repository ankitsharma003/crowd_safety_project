# movement_instruction.py
# Generate movement instructions and visualize flow arrows on anomaly maps

import numpy as np
from PIL import Image, ImageDraw
from flow_estimation import load_images, estimate_flow, dominant_direction, dominant_direction_from_pair, estimate_shifts_confidence
from venue_config import nearest_exit, which_zone
import math
from bottleneck_detection import detect_bottlenecks

def draw_flow_arrow(img, flow):
    # Draw a simple arrow indicating dominant flow direction
    h, w = img.size
    draw = ImageDraw.Draw(img)
    # Compute average movement in x and y
    img_np = np.array(img, dtype=np.float32) / 255.0
    y_indices, x_indices = np.mgrid[0:h, 0:w]
    total = np.sum(img_np)
    if total == 0:
        return img
    x_flow = np.sum(img_np * x_indices) / total - w/2
    y_flow = np.sum(img_np * y_indices) / total - h/2
    start = (w//2, h//2)
    end = (int(w//2 + x_flow), int(h//2 + y_flow))
    draw.arrow([start, end], fill='red', width=3)
    return img

def _centroid_of_density(image_2d: np.ndarray) -> tuple:
    h, w = image_2d.shape
    y_indices, x_indices = np.mgrid[0:h, 0:w]
    total = float(np.sum(image_2d))
    if total <= 0.0:
        return (w/2.0, h/2.0)
    cx = float(np.sum(image_2d * x_indices) / total)
    cy = float(np.sum(image_2d * y_indices) / total)
    return (cx, cy)

def _angle_to_direction(dx: float, dy: float) -> str:
    angle = math.degrees(math.atan2(dy, dx))
    if -45 <= angle < 45:
        return "East"
    if 45 <= angle < 135:
        return "South"
    if -135 <= angle < -45:
        return "North"
    return "West"

def generate_instructions(images, flows, bottlenecks):
    instructions = []
    for i in range(len(images)):
        if i == 0 or i >= len(flows):
            instructions.append("No flow data.")
            continue
        flow = flows[i-1]
        direction = dominant_direction(flow)
        # Prefer shift-based estimate for robustness
        try:
            direction = dominant_direction_from_pair(images[i-1], images[i])
        except Exception:
            pass
        # Compute density centroid and decide nearest exit guidance
        cx, cy = _centroid_of_density(images[i])
        exit_name = nearest_exit(cx, cy)
        zone_name = which_zone(cx, cy)
        guidance = f"Use {exit_name} from {zone_name}."
        if i in bottlenecks:
            # Suggest alternative exit if bottleneck persists
            instructions.append(f"Bottleneck detected! {direction} Avoid current area; {guidance}")
        else:
            instructions.append(f"Flow: {direction} {guidance}")
    return instructions

def generate_instruction_details(images, flows, bottlenecks, alpha: float = 0.75):
    """
    Returns a list of dicts with structured guidance per frame:
    {
        'frame': int,
        'bottleneck': bool,
        'flow_text': str,        # e.g., "Suggest moving right (East)"
        'direction_cardinal': str, # East/West/North/South
        'exit': str,             # Exit A/B/C/D
        'zone': str,             # Zone name
        'recommendation': str,   # concise user-facing message
        'severity': str          # 'info' | 'warning'
    }
    """
    details = []
    shifts = estimate_shifts_confidence(images, alpha=alpha)
    for i in range(len(images)):
        if i == 0 or i >= len(flows):
            details.append({
                'frame': i,
                'bottleneck': False,
                'flow_text': 'No movement detected',
                'direction_cardinal': 'None',
                'exit': 'N/A',
                'zone': 'N/A',
                'recommendation': 'Insufficient data for this frame.',
                'confidence': 0.0,
                'severity': 'info',
            })
            continue
        flow = flows[i-1]
        # Use smoothed global shifts for consistent direction + confidence gating
        conf = shifts[i-1]['confidence'] if (i-1) < len(shifts) else 0.0
        dx = shifts[i-1]['dx'] if (i-1) < len(shifts) else 0.0
        dy = shifts[i-1]['dy'] if (i-1) < len(shifts) else 0.0
        mag = math.hypot(dx, dy)
        CONF_MIN = 0.25
        MAG_MIN = 0.5
        if conf < CONF_MIN or mag < MAG_MIN:
            flow_text = 'No movement detected'
            cardinal = 'None'
        else:
            if abs(dx) > abs(dy):
                flow_text = 'Suggest moving right (East)' if dx > 0 else 'Suggest moving left (West)'
            else:
                flow_text = 'Suggest moving down (South)' if dy > 0 else 'Suggest moving up (North)'
            cardinal = _angle_to_direction(dx, dy)
        # centroid & nearest exit
        cx, cy = _centroid_of_density(images[i])
        exit_name = nearest_exit(cx, cy)
        zone_name = which_zone(cx, cy)
        is_bottleneck = i in bottlenecks
        severity = 'warning' if is_bottleneck else 'info'
        if is_bottleneck:
            recommendation = f"Bottleneck detected near {zone_name}. Detour: proceed towards {exit_name}."
        else:
            if cardinal == 'None':
                recommendation = f"Proceed towards {exit_name} from {zone_name}."
            else:
                recommendation = f"Proceed {cardinal} towards {exit_name} from {zone_name}."
        details.append({
            'frame': i,
            'bottleneck': is_bottleneck,
            'flow_text': flow_text,
            'direction_cardinal': cardinal,
            'exit': exit_name,
            'zone': zone_name,
            'recommendation': recommendation,
            'confidence': round(float(conf), 3),
            'severity': severity,
        })
    return details

if __name__ == "__main__":
    folder = r"UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test001"
    images = load_images(folder)
    flows = estimate_flow(images)
    bottlenecks = detect_bottlenecks(images, flows)
    instructions = generate_instructions(images, flows, bottlenecks)
    for i, instr in enumerate(instructions):
        print(f"Frame {i}: {instr}")
    # Optionally, visualize flow arrows on images
    for i in bottlenecks:
        img = Image.fromarray((images[i]*255).astype(np.uint8))
        img = img.resize((128,128))
        img_arrow = draw_flow_arrow(img, flows[i-1] if i > 0 else np.zeros_like(images[0]))
        img_arrow.save(f"bottleneck_arrow_{i:03d}.png")
    print("Flow arrow images saved for bottleneck frames.")
