import json
import yaml
import numpy as np
from pathlib import Path
from src.models import ObjectStats
from src.object_stats import ObjectStatsManager
from src.homography import Homography
from src.heatmap import HeatmapRenderer
from src.output_writer import OutputWriter

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def rebuild_from_jsonl(jsonl_path: str, config: dict) -> ObjectStatsManager:
    homography = Homography(config["homography"])
    homography.load(config["homography"]["matrix_path"])

    stats_manager = ObjectStatsManager(config, homography)

    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line.strip())
            if not record:
                continue

            track_id = record["track_id"]

            # get or create the stats object for this track
            if track_id not in stats_manager._stats:
                stats_manager._stats[track_id] = ObjectStats(
                    track_id   = track_id,
                    # class_name = record.get("class_name", "player"),
                )

            stats = stats_manager._stats[track_id]

            # restore identity
            stats.name = record.get("name", "unknown")
            stats.team = record.get("team", -1)

            # restore positions
            stats.positions.append((
                record["frame"],
                record["pitch_x"],
                record["pitch_y"],
            ))

            # restore speeds + accelerations
            speed_ms = record.get("speed_ms", 0.0)
            accel    = record.get("acceleration", 0.0)
            stats.speeds.append(speed_ms)
            stats.accelerations.append(accel)

            # restore distance + max speed
            stats.distance_covered = max(
                stats.distance_covered,
                record.get("pitch_x", 0.0)   # will be recalculated below
            )
            stats.max_speed = max(stats.max_speed, speed_ms)

            # rebuild heatmap bin
            from src.homography import PitchPoint
            pos = PitchPoint(x_m=record["pitch_x"], y_m=record["pitch_y"])
            col, row = homography.to_grid_cell(pos)
            h, w = stats.heatmap.shape
            if 0 <= row < h and 0 <= col < w:
                stats.heatmap[row, col] += 1

    # recalculate distance_covered properly from positions
    for stats in stats_manager._stats.values():
        total = 0.0
        for i in range(1, len(stats.positions)):
            _, x0, y0 = stats.positions[i-1]
            _, x1, y1 = stats.positions[i]
            total += np.hypot(x1 - x0, y1 - y0)
        stats.distance_covered = total

    return stats_manager


def main():
    config = load_config()

    jsonl_path = Path(config["output"]["output_dir"]) / \
                 f"{config['output']['match_id']}_frames.jsonl"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Could not find {jsonl_path}")

    print(f"Reading from {jsonl_path}...")
    stats_manager = rebuild_from_jsonl(str(jsonl_path), config)
    print(f"Rebuilt stats for {len(stats_manager._stats)} tracks.")

    heatmap_renderer = HeatmapRenderer(config["homography"])

    # write fresh summary
    with OutputWriter(config["output"]) as writer:
        writer.write_summary(stats_manager, heatmap_renderer)

    print("Done.")


if __name__ == "__main__":
    main()