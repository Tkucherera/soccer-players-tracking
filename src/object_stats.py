import numpy as np
from collections import defaultdict
from .models import ObjectStats, Track, PitchPoint, FrameMetadata
from .homography import Homography


SPEED_ZONES = {
    "standing":  (0,    0.5),
    "walking":   (0.5,  2.0),
    "jogging":   (2.0,  4.0),
    "running":   (4.0,  6.0),
    "sprinting": (6.0,  float("inf")),
}

class ObjectStatsManager:
    def __init__(self, config: dict, homography: Homography):
        self.fps = config.get("fps", 30)
        self.homography = homography
        self.grid_size = config.get("grid_size_m", 5.0) # size of heatmap grid cells in metres

        # one ObjectStats instance per track_id
        self._stats: dict[int, ObjectStats] = {}

    def get_or_create_stats(self, track: Track) -> ObjectStats:
        if track.track_id not in self._stats:
            self._stats[track.track_id] = ObjectStats(track_id=track.track_id, name=track.class_name)
        return self._stats[track.track_id]
    
    def update(self, track: Track, pitch_pos: PitchPoint, meta: FrameMetadata):
        stats = self.get_or_create_stats(track)

        stats.positions.append((meta.frame_number, pitch_pos.x_m, pitch_pos.y_m))

        if len(stats.positions) > 1:
            _, x1, y1 = stats.positions[-2]
            _, x2, y2 = stats.positions[-1]

            dx = x2 - x1
            dy = y2 - y1
            distance = np.sqrt(dx**2 + dy**2)
            speed     = distance * self.fps
            direction = np.arctan2(dy, dx)

            stats.speeds.append(speed)                    # plain float
            stats.directions.append((meta.frame_number, direction))  # tuple ok, not used in maths

            if len(stats.speeds) > 1:
                prev_speed   = stats.speeds[-2]           # plain float now
                acceleration = (speed - prev_speed) * self.fps
                stats.accelerations.append(acceleration)  # plain float

            stats.distance_covered += distance
            stats.max_speed = max(stats.max_speed, speed)

        col, row = self.homography.to_grid_cell(pitch_pos)
        stats.heatmap[row, col] += 1.0

    def all_stats(self) -> dict[int, ObjectStats]:
        return self._stats
    
    def get(self, track_id: int) -> ObjectStats | None:
        return self._stats.get(track_id)
    
    def smoothed_speed(self, track_id: int, window: int = 5) -> list[float]:
        speeds = self._stats[track_id].speeds
        if len(speeds) < window:
            return speeds
        kernel = np.ones(window) / window
        return np.convolve(speeds, kernel, mode="valid").tolist()

    def smoothed_accel(self, track_id: int, window: int = 5) -> list[float]:
        accels = self._stats[track_id].accelerations
        if len(accels) < window:
            return accels
        kernel = np.ones(window) / window
        return np.convolve(accels, kernel, mode="valid").tolist()
    
    def normalised_heatmap(self, track_id: int) -> np.ndarray:
        h = self._stats[track_id].heatmap
        max_val = h.max()
        if max_val == 0:
            return h
        return h / max_val
    
    def speed_zone_breakdown(self, track_id: int) -> dict[str, float]:
        """Returns seconds spent in each speed zone."""
        stats = self._stats[track_id]
        fps   = self.fps
        zones = {z: 0.0 for z in SPEED_ZONES}

        for speed in stats.speeds:
            for zone, (low, high) in SPEED_ZONES.items():
                if low <= speed < high:
                    zones[zone] += 1 / fps    # frames → seconds
                    break

        return zones

    def summary(self, track_id: int) -> dict:
        stats = self._stats[track_id]
        return {
            "track_id":        stats.track_id,
            "name":            stats.name,
            "team":            stats.team,
            "class_name":      stats.class_name,
            "distance_m":      round(stats.distance_covered, 2),
            "max_speed_ms":    round(stats.max_speed, 2),
            "max_speed_kmh":   round(stats.max_speed * 3.6, 2),
            "speed_zones":     self.speed_zone_breakdown(track_id),
            "heatmap":         self.normalised_heatmap(track_id).tolist(),
            "events":          stats.events,
            "frames_tracked":  len(stats.positions),
        }

    def all_summaries(self) -> list[dict]:
        return [self.summary(tid) for tid in self._stats]