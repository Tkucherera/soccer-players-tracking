from dataclasses import dataclass, field
import numpy as np


CLASS_NAMES = {0: "player", 1: "referee", 2: "ball"}

@dataclass
class Detection:
    bbox: tuple[int, int, int, int] # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str

@dataclass
class DetectionOutput:
    objects: list[Detection] # players + referee -> go to tracker
    ball: Detection | None # handled reparately


@dataclass
class Track:
    track_id: int
    bbox: tuple[int, int, int, int] # (x1, y1, x2, y2)
    class_id: int
    class_name: str
    confidence: float
    is_confirmed: bool = False  # becomes True after a few frames of consistent detection

@dataclass
class ObjectStats:
    track_id: int
    name: str = "unknown"
    team: int | None = None
    class_name: str = "player"
    positions: list = field(default_factory=list)  # list of (x_center, y_center) over time
    speeds: list = field(default_factory=list)     # list of speeds over time
    directions: list = field(default_factory=list) # list of movement directions over time
    accelerations: list = field(default_factory=list) # list of accelerations over time
    heatmap: np.ndarray | None = field(default_factory=lambda: np.zeros((68, 105), dtype=np.float32)) # 2D heatmap of presence on the pitch

    distance_covered: float = 0.0
    max_speed: float = 0.0
    events: list = field(default_factory=list) # e.g., "pass", "shot", "


@dataclass
class PitchPoint:
    x_m : float # meters from top left corner of the pitch
    y_m : float

@dataclass
class FrameMetadata:
    frame_number: int
    timestamp_s: float
    width: int
    height: int

@dataclass
class Event:
    event_type: str
    frame: int
    timestamp_s: float
    primary_track_id: int # player initiating the event
    secondary_track_id: int | None = None # opponent or teammate involved, if applicable
    ball_position: tuple[float, float] | None = None # (x_m, y_m) on the pitch at the time of the event
    confidence: float = 0.0       