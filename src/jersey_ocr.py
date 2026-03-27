import re
import cv2
import numpy as np 
import logging 
from paddleocr import PaddleOCR
from .models import Track

log = logging.getLogger(__name__)


"""
This Jersey OCR is not quite there yet and frankly might require a lot of 
work both on the video input quality and ocr itself 

For now the best workaround for now is to just actually just put
the name of player to the track. 

Plainly this Jersey OCR might just be an waistful step and we
will continue to work on it to make it better 
"""

class JerseyOCR:
    def __init__(self, config: dict):
        self.run_every_n = config.get("run_every_n_frames", 10)
        self.min_confirms = config.get("min_confirmations", 3)

        # PaddleOCR English, no angle classification (jerseys are upright)
        self.ocr = PaddleOCR(
            use_angle_cls = False,
            lang="en",
            # show_log=False,
        )

        # pending confirmations before committing 
        # { track_id: { jersey_number: count } }
        self._candidates: dict[int, dict[int, int]] = {}


    def update(self, track: Track, frame: np.ndarray, registry: dict):
        """
        Called every N frames per track.
        Writes to registry once a number is confirmed 
        registry format: { track_id: { "jersey_number": int, "confirmed": bool } }

        """
        # skip if already confirmed 
        entry = registry.get(track.track_id)
        if entry and entry.get("confirmed"):
            return
        
        crop = self._crop_torso(frame, track.bbox)
        if crop is None:
            return 
        
        number = self._read_number(crop)
        if number is None:
            return
        
        # accumulate candidate votes 
        if track.track_id not in self._candidates:
            self._candidates[track.track_id] = {}

        counts = self._candidates[track.track_id]
        counts[number] = counts.get(number, 0) + 1

        log.debug(f"Track {track.track_id} - Jersey candidate {number}"
                  f"({counts[number]}/{self.min_confirms})")
        
        # commit once a number reaches min_confirmations 
        best = max(counts, key=counts.get)
        if counts[best] >= self.min_confirms:
            registry[track.track_id] = {
                "jersey_number": best,
                "confirmed": True,
            }
            log.info(f"Track {track.track_id} confirmed as jersey #{best}")

    def _crop_torso(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """
        Crops the middle third of the bounding box vertically —
        avoids the head (top third) and legs (bottom third)
        where numbers never appear.
        """
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        # need a minimum size to be worth running OCR on
        if h < 30 or w < 15:
            return None

        torso_y1 = y1 + h // 3
        torso_y2 = y1 + (2 * h) // 3

        # clamp to frame bounds
        fh, fw = frame.shape[:2]
        torso_y1 = max(0, torso_y1)
        torso_y2 = min(fh, torso_y2)
        x1       = max(0, x1)
        x2       = min(fw, x2)

        crop = frame[torso_y1:torso_y2, x1:x2]

        if crop.size == 0:
            return None

        return self._preprocess(crop)
    
    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
            """
            Upscale and increase contrast — small bounding boxes at
            broadcast resolution are too low-res for OCR without this.
            """
            # upscale 3x so digits are large enough to recognise
            scale = 3
            crop = cv2.resize(crop,
                            (crop.shape[1] * scale, crop.shape[0] * scale),
                            interpolation=cv2.INTER_CUBIC)

            # convert to grayscale + apply CLAHE for contrast
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            gray = clahe.apply(gray)

            # back to BGR so PaddleOCR is happy
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _read_number(self, crop: np.ndarray) -> int | None:
        """
        Runs PaddleOCR and extracts the first 1-2 digit number found.
        Returns None if nothing useful is detected.
        """
        try:
            result = self.ocr.ocr(crop, cls=False)
        except Exception as e:
            log.debug(f"OCR error: {e}")
            return None

        if not result or not result[0]:
            return None

        for line in result[0]:
            text, confidence = line[1]

            if confidence < 0.6:
                continue

            # strip everything except digits
            digits = re.sub(r"\D", "", text)

            if not digits:
                continue

            # jersey numbers are 1–99
            number = int(digits[:2])   # take first two digits max
            if 1 <= number <= 99:
                return number

        return None