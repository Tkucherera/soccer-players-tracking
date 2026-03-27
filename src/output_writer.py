import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime
from .models import ObjectStats

class OutputWriter:

    def __init__(self, config: dict):
        self.out_dir     = Path(config["output_dir"])
        self.match_id    = config.get("match_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.write_jsonl = config.get("write_jsonl", True)
        self.write_csv   = config.get("write_csv", True)

        self.out_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path   = self.out_dir / f"{self.match_id}_frames.jsonl"
        self._summary_path = self.out_dir / f"{self.match_id}_summary.json"
        self._csv_path     = self.out_dir / f"{self.match_id}_frames.csv"

        self._jsonl_file = None
        self._csv_writer = None
        self._csv_file   = None

    def open(self):
        """Call once before the pipeline loop starts."""
        if self.write_jsonl:
            self._jsonl_file = open(self._jsonl_path, "w")

        if self.write_csv:
            self._csv_file   = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=[
                "match_id", "frame", "timestamp_s",
                "track_id", "name", "team",
                "pitch_x", "pitch_y",
                "speed_ms", "speed_kmh", "acceleration",
                "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            ])
            self._csv_writer.writeheader()

    def write_frame(self, track, pitch_pos, stats, meta):
        raw_speed = stats.speeds[-1] if stats.speeds else 0.0
        raw_accel = stats.accelerations[-1] if stats.accelerations else 0.0

        # unwrap tuple if stats are stored as (frame, value)
        speed_ms = raw_speed[1] if isinstance(raw_speed, tuple) else raw_speed
        accel    = raw_accel[1] if isinstance(raw_accel, tuple) else raw_accel

        # base record — no bbox key here
        record = {
            "match_id":     self.match_id,
            "frame":        meta.frame_number,
            "timestamp_s":  round(meta.timestamp_s, 3),
            "track_id":     track.track_id,
            "name":         stats.name,
            "team":         stats.team,
            "pitch_x":      round(pitch_pos.x_m, 3),
            "pitch_y":      round(pitch_pos.y_m, 3),
            "speed_ms":     round(speed_ms, 3),
            "speed_kmh":    round(speed_ms * 3.6, 3),
            "acceleration": round(accel, 3),
        }

        if self.write_jsonl and self._jsonl_file:
            # JSONL can have bbox as a list — no fieldname constraint
            self._jsonl_file.write(json.dumps({**record, "bbox": list(track.bbox)}) + "\n")

        if self.write_csv and self._csv_writer:
            # CSV gets the four individual columns instead
            self._csv_writer.writerow({
                **record,
                "bbox_x1": track.bbox[0],
                "bbox_y1": track.bbox[1],
                "bbox_x2": track.bbox[2],
                "bbox_y2": track.bbox[3],
            })
    
    def write_summary(self, stats_manager, heatmap_renderer):
        """
        Called once at the end of the match.
        Writes one JSON file with all player summaries + heatmaps.
        """
        summaries = []

        for track_id, stats in stats_manager.all_stats().items():
            summary = {
                "match_id":      self.match_id,
                "track_id":      track_id,
                "name":          stats.name,
                "team":          stats.team,
                "class_name":    stats.class_name,
                "distance_m":    round(stats.distance_covered, 2),
                "max_speed_ms":  round(stats.max_speed, 2),
                "max_speed_kmh": round(stats.max_speed * 3.6, 2),
                "frames_tracked": len(stats.positions),
                "speed_zones":   self._speed_zones(stats),
                "events":        stats.events,
                "heatmap":       heatmap_renderer.to_json(stats.heatmap),
            }
            summaries.append(summary)

        # team heatmaps
        all_stats = list(stats_manager.all_stats().values())
        team_a = [s for s in all_stats if s.team == 0]
        team_b = [s for s in all_stats if s.team == 1]

        output = {
            "match_id":        self.match_id,
            "generated_at":    datetime.now().isoformat(),
            "player_summaries": summaries,
            "team_heatmaps": {
                "team_a": heatmap_renderer.to_json(
                              heatmap_renderer.team_heatmap(team_a)),
                "team_b": heatmap_renderer.to_json(
                              heatmap_renderer.team_heatmap(team_b)),
            }
        }

        with open(self._summary_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Summary written to {self._summary_path}")

    def close(self):
        """Call after the pipeline loop ends."""
        if self._jsonl_file:
            self._jsonl_file.flush()
            self._jsonl_file.close()
        if self._csv_file:
            self._csv_file.flush()
            self._csv_file.close()

    def _speed_zones(self, stats: ObjectStats) -> dict:
        zones = {
            "standing":  0.0,
            "walking":   0.0,
            "jogging":   0.0,
            "running":   0.0,
            "sprinting": 0.0,
        }
        thresholds = [
            ("standing",  0,   0.5),
            ("walking",   0.5, 2.0),
            ("jogging",   2.0, 4.0),
            ("running",   4.0, 6.0),
            ("sprinting", 6.0, float("inf")),
        ]
        fps = len(stats.speeds) / max(len(stats.positions), 1) * 25
        for speed in stats.speeds:
            for zone, low, high in thresholds:
                if low <= speed < high:
                    zones[zone] += round(1 / fps, 4)
                    break
        return zones

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()