"""
Venue configuration: exits and zones for guidance.

Coordinates are defined on the 64x64 image grid used by the pipeline.
"""

from typing import List, Tuple, Dict

# Exits are labeled points on the boundary of the 64x64 grid
EXITS: Dict[str, Tuple[int, int]] = {
    "Exit A": (32, 0),     # top-center
    "Exit B": (63, 32),    # right-center
    "Exit C": (32, 63),    # bottom-center
    "Exit D": (0, 32),     # left-center
}

# Zones are rectangular regions: (name, x_min, y_min, x_max, y_max)
ZONES: List[Tuple[str, int, int, int, int]] = [
    ("Zone A", 0, 0, 21, 21),
    ("Zone B", 42, 0, 63, 21),
    ("Zone C", 0, 42, 21, 63),
    ("Zone D", 42, 42, 63, 63),
]

def nearest_exit(x: float, y: float) -> str:
    best_name = "Exit A"
    best_dist2 = float("inf")
    for name, (ex, ey) in EXITS.items():
        dx = ex - x
        dy = ey - y
        d2 = dx*dx + dy*dy
        if d2 < best_dist2:
            best_dist2 = d2
            best_name = name
    return best_name

def which_zone(x: float, y: float) -> str:
    for name, x0, y0, x1, y1 in ZONES:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return "Center"


