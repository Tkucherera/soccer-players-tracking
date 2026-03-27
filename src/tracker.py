from dataclasses import dataclass
import numpy as np
import logging
from boxmot import BotSort
from pathlib import Path
from .models import Track, Detection, CLASS_NAMES


logging.basicConfig(level=logging.INFO)


class Tracker:
    def __init__(self, config: dict):
        self.tracker = BotSort(
            reid_weights=Path(config["reid_model_path"]),
            device=config.get("device", "cpu"),
            half=False,
        )
        self.min_hits = config.get("min_hits", 3)  # frames to confirm a track

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        if not detections:
            # still need to call update to maintain tracks even if no detections
            return self._parse(self.tracker.update(np.empty((0, 6)), frame))
        
        # BotSort expects detections in the format [x1, y1, x2, y2, confidence, class_id]
        det_array = np.array([
            [*det.bbox, det.confidence, det.class_id] for det in detections
        ], dtype=np.float32)

        raw_tracks = self.tracker.update(det_array, frame)
        return self._parse(raw_tracks)

    def _parse(self, raw_tracks: np.ndarray) -> list[Track]:
        tracks = []
        for row in raw_tracks:
            x1, y1, x2, y2, track_id, conf, class_id, *_ = row
            class_id = int(class_id)
            tracks.append(Track(
                track_id=int(track_id),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                class_id=class_id,
                class_name=CLASS_NAMES.get(class_id, "unknown"),
                confidence=float(conf),
                is_confirmed=True  # BotSort only returns confirmed tracks
            ))
        return tracks