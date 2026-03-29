import cv2
import yaml
import logging
from pathlib import Path

from src.detector import Detector
from src.pitch_mask import PitchMasker
from src.video_source import VideoSource
from src.homography import Homography, calibrate_from_frame
from src.object_stats import ObjectStatsManager
from src.tracker import Tracker
from src.event_detector import EventDetector
from src.output_writer  import OutputWriter
from src.heatmap import HeatmapRenderer
from src.visualiser import Visualiser
from src.jersey_ocr import JerseyOCR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── modules ───────────────────────────────────────────────────────────────────
pitch_masker     = PitchMasker(config["pitch_masker"])
detector         = Detector(config["detector"])
video_source     = VideoSource(config["video_source"], config)
homography       = Homography(config["homography"])
stats_manager    = ObjectStatsManager(config["stats"], homography)
tracker          = Tracker(config["tracker"])
event_detector   = EventDetector(config["event_detector"], stats_manager)
heatmap_renderer = HeatmapRenderer(config["heat_map"])
visualiser       = Visualiser(config["visualiser"], homography)
jersey_ocr = JerseyOCR(config["jersey_ocr"])
registry = {}

# ── homography calibration ────────────────────────────────────────────────────
first_frame, _ = next(video_source)
video_source.reset()

matrix_path = Path(config["homography"]["matrix_path"])
if matrix_path.exists():
    log.info(f"Loading homography matrix from {matrix_path}")
    homography.load(str(matrix_path))
else:
    log.info("No matrix found — starting interactive calibration...")
    calibrate_from_frame(homography, first_frame)
    homography.save(str(matrix_path))
    log.info(f"Matrix saved to {matrix_path}")

# ── pipeline loop ─────────────────────────────────────────────────────────────
log.info("Starting pipeline...")
frame_count = 0

with OutputWriter(config["output"]) as writer:
    for frame, meta in video_source:

        # 1. mask + detect
        mask   = pitch_masker.run(frame)
        output = detector.run(frame, mask)

        # 2. track
        tracks = tracker.update(output.objects, frame)

        # 3. pitch coordinates for all active tracks
        pitch_positions = {}
        for track in tracks:
            foot = Detector.footpoint(track.bbox) 
            pos  = homography.pixel_to_pitch(*foot)
            if pos:
                pitch_positions[track.track_id] = pos

        # 4. accumulate stats + write per-frame records
        for track in tracks:
            if track.track_id not in pitch_positions:
                continue
            pos = pitch_positions[track.track_id]
            stats_manager.update(track, pos, meta)

            # try out this jersey ocr
            # if meta.frame_number % config["jersey_ocr"]["run_every_n_frames"] == 0:
            #    jersey_ocr.update(track, frame, registry)

            writer.write_frame(
                track, pos, stats_manager.get(track.track_id), meta
            )

        # 5. event detection
        event_detector.update(tracks, output.ball, pitch_positions, meta)

        # 6. visualise
        annotated = visualiser.draw(
            frame, tracks, stats_manager.all_stats(), output.ball
        )
        key = visualiser.show(annotated)
        if key == 27:   # ESC to stop early
            log.info("Stopped early by user.")
            break

        frame_count += 1
        if frame_count % 100 == 0:
            log.info(
                f"Frame {frame_count} | "
                f"{meta.timestamp_s:.1f}s | "
                f"{len(tracks)} tracks | "
                f"{len(pitch_positions)} on pitch"
            )

    # 7. end of match summary
    log.info("Writing summary...")
    writer.write_summary(stats_manager, heatmap_renderer)

summaries = stats_manager.all_summaries()
log.info(f"Done — {len(summaries)} player summaries generated.")