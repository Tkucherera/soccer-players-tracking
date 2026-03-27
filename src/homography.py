"""
is the module that converts pixel coordinates from the camera frame into 
real-world metres on the pitch. Without it, everything is in pixels 
you can't compute speed in m/s, distance in metres, or place a player 
accurately on a heatmap. It's the bridge between the computer vision 
world and the physical world.

The core idea

A homography is a 3×3 matrix H that maps any point in the camera image to a 
corresponding point on a flat plane — in this case the pitch surface. You 
compute it once from a set of known correspondences (pixel positions of 
pitch markings whose real-world coordinates you know), then apply it to 
every footpoint every frame.
"""


import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from .models import PitchPoint

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


KNOWN_WORLD_POINTS = {
    "top_left_corner":         (0,    0),
    "top_right_corner":        (105,  0),
    "bottom_left_corner":      (0,   68),
    "bottom_right_corner":     (105, 68),
    "centre_spot":             (52.5, 34),
    "left_penalty_spot":       (11,  34),
    "right_penalty_spot":      (94,  34),
    "top_left_penalty_box":    (0,   13.84),
    "top_right_penalty_box":   (105, 13.84),
    "bottom_left_penalty_box": (0,   54.16),
}


class Homography:
    def __init__(self, config: dict):
        self.pitch_length = config.get("pitch_length_m", 105.0) # in metres
        self.pitch_width = config.get("pitch_width_m", 68.0)    # in metres
        self.grid_size = config.get("grid_size_m", 2.0)              # size of grid cells in metres

        # H is none untile calibrate is called
        self.H: np.ndarray | None  = None

    def calibrate(self, pixel_points: list[tuple], world_points: list[tuple]):
        if len(pixel_points) < 4:
            raise ValueError("At least 4 point correspondences required.")
        if len(pixel_points) < 6:
            log.warning("Only %d calibration points — accuracy may be poor "
                        "toward unseen pitch areas. 6+ recommended.", 
                        len(pixel_points))

        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)
        self.H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

        if self.H is None:
            raise ValueError("Homography calibration failed — check your point pairs.")

    def pixel_to_pitch(self, px: int, py: int) -> PitchPoint | None:
        if self.H is None:
            raise RuntimeError("Call calibrate() before pixel_to_pitch()")

        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self.H)
        x_m, y_m = result[0][0]

        # sanity check — discard if outside pitch bounds
        if not (0 <= x_m <= self.pitch_length and 0 <= y_m <= self.pitch_width):
            return None

        return PitchPoint(x_m=float(x_m), y_m=float(y_m))

    def to_grid_cell(self, point: PitchPoint) -> tuple[int, int]:
        """Returns (col, row) grid cell index for heatmap binning."""
        col = int(point.x_m / self.grid_size)
        row = int(point.y_m / self.grid_size)
        return (col, row)
    
    def save(self, path: str):
        np.save(path, self.H.astype(np.float64))

    def load(self, path: str):

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"No homography matrix at {path} — run calibration first")
        self.H = np.load(path).astype(np.float32)
      
NAMED_WORLD_POINTS = [
    ((0,    0),    "top-left corner"),
    ((105,  0),    "top-right corner"),
    ((0,   68),    "bottom-left corner"),
    ((105, 68),    "bottom-right corner"),
    ((52.5, 0),    "top half-way line"),
    ((52.5, 68),   "bottom half-way line"),
    ((52.5, 34),   "centre spot"),
    ((0,   13.84), "top-left penalty box"),
    ((105, 13.84), "top-right penalty box"),
    ((0,   54.16), "bottom-left penalty box"),
    ((105, 54.16), "bottom-right penalty box"),
    ((16.5, 0),    "top-left 18-yard box side"),
    ((88.5, 0),    "top-right 18-yard box side"),
    ((11,  34),    "left penalty spot"),
    ((94,  34),    "right penalty spot"),
]

def calibrate_from_frame(homography: Homography, frame: np.ndarray):
    """
    Click any visible pitch markings in any order.
    A menu lets you select which point you are clicking.
    Press D when done (minimum 4 points needed).
    Press U to undo the last point.
    """
    selected_world = []
    selected_pixel = []
    current_index  = [0]    # which point in the menu is highlighted
    display        = [frame.copy()]

    def redraw():
        img = frame.copy()
        # draw already-clicked points
        for i, (px, py) in enumerate(selected_pixel):
            cv2.circle(img, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(img, str(i + 1), (px + 8, py - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        # draw menu in top-left
        y = 20
        for i, (_, label) in enumerate(NAMED_WORLD_POINTS):
            already = any(w == NAMED_WORLD_POINTS[i][0] for w in selected_world)
            colour = (180, 180, 180) if already else \
                     (0, 255, 255)   if i == current_index[0] else \
                     (255, 255, 255)
            prefix = "  [done] " if already else \
                     ">> "        if i == current_index[0] else \
                     "   "
            cv2.putText(img, f"{prefix}{label}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1)
            y += 18
        cv2.putText(img,
                    "UP/DOWN: select point  |  CLICK: place  |  U: undo  |  D: done",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
        display[0] = img
        cv2.imshow("Calibration", img)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            world_pt = NAMED_WORLD_POINTS[current_index[0]][0]
            # don't allow the same world point twice
            if world_pt in selected_world:
                return
            selected_pixel.append((x, y))
            selected_world.append(world_pt)
            print(f"  Point {len(selected_pixel)}: "
                  f"pixel ({x},{y}) → world {world_pt}  "
                  f"({NAMED_WORLD_POINTS[current_index[0]][1]})")
            redraw()

    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", on_click)
    redraw()

    calibrated = [False]

    while not calibrated[0]:
        key = cv2.waitKey(50)

        if key == 27:    # ESC
            raise RuntimeError("Calibration aborted.")

        elif key == ord('d') or key == ord('D'):
            if len(selected_pixel) < 4:
                print(f"  Need at least 4 points, only have {len(selected_pixel)}. Keep clicking.")
            else:
                homography.calibrate(selected_pixel, selected_world)
                print(f"Homography calibrated with {len(selected_pixel)} points.")
                calibrated[0] = True

        elif key == ord('u') or key == ord('U'):
            if selected_pixel:
                removed_px = selected_pixel.pop()
                removed_w  = selected_world.pop()
                print(f"  Undid point: pixel {removed_px} → world {removed_w}")
                redraw()

        elif key == 82 or key == ord('w'):   # UP arrow or W
            current_index[0] = (current_index[0] - 1) % len(NAMED_WORLD_POINTS)
            redraw()

        elif key == 84 or key == ord('s'):   # DOWN arrow or S
            current_index[0] = (current_index[0] + 1) % len(NAMED_WORLD_POINTS)
            redraw()

    cv2.waitKey(500)
    cv2.destroyAllWindows()
    cv2.waitKey(1)# flush the destroy event through the event loop