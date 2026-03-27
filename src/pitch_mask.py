import cv2
import numpy as np


"""
This works for standard pitch if its snowing the mask might need to be inverted (white is pitch, black is non-pitch).

"""


# This tuning is for broadcast football, might need to adjust 
GRASS_LOWER = np.array([30, 40, 40])  # Lower HSV bound for grass
GRASS_UPPER = np.array([85, 255, 255]) # Upper HSV

class PitchMasker:
    def __init__(self, config: dict):
        lower = config.get("grass_hsv_lower", GRASS_LOWER)
        upper = config.get("grass_hsv_upper", GRASS_UPPER)

        self.lower = np.array(lower, dtype=np.uint8)
        self.upper = np.array(upper, dtype=np.uint8)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))


    def run(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)

        # close small holes in the mask (shadows, lines, worn patches)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

        # remove isolated specks outside the pitch area
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel) # remove small noise
        return mask

def calibrate(frame: np.ndarray):
    """
    Run this once on a sample frame to find your HSV range.
    Click pixels on the grass in the displayed window. This one 
    might actually need to change once the video source changes
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"HSV at ({x},{y}): {hsv[y, x]}")

    cv2.imshow("calibrate", frame)
    cv2.setMouseCallback("calibrate", on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()