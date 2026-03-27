import cv2
import numpy as np
from .models import Track, ObjectStats
from .homography import Homography, PitchPoint

class Visualiser:

    def __init__(self, config: dict, homography: Homography):
        self.homography    = homography
        self.pitch_length  = config.get("pitch_length_m", 105)
        self.pitch_width   = config.get("pitch_width_m", 68)
        self.show_speed    = config.get("show_speed", True)
        self.show_name     = config.get("show_name", True)
        self.show_minimap  = config.get("show_minimap", True)
        self.minimap_scale = config.get("minimap_scale", 6)   # pixels per metre
        self.write_video   = config.get("write_video", False)
        self.output_path   = config.get("output_path", "output/annotated.mp4")

        # team colours — BGR
        self.team_colours = {
            0:  (235, 120,  30),   # team A — blue
            1:  ( 30, 120, 235),   # team B — red
            -1: (180, 180, 180),   # unknown — gray
        }

        self._writer = None   # cv2.VideoWriter, opened on first frame

    def _ensure_writer(self, frame: np.ndarray):
        if self._writer is not None:
            return
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.output_path, fourcc, 25, (w, h)
        )

    def draw(self,
             frame: np.ndarray,
             tracks: list[Track],
             stats_map: dict[int, ObjectStats],
             ball=None) -> np.ndarray:
        """
        Returns an annotated copy of the frame.
        Does not modify the original.
        """
        out = frame.copy()

        for track in tracks:
            stats = stats_map.get(track.track_id)
            colour = self.team_colours.get(
                stats.team if stats else -1, (180, 180, 180)
            )
            self._draw_player(out, track, stats, colour)

        if ball is not None:
            self._draw_ball(out, ball)

        if self.show_minimap:
            out = self._draw_minimap(out, tracks, stats_map, ball)

        if self.write_video and self._writer:
            self._writer.write(out)

        return out

    def _draw_player(self, frame, track, stats, colour):
        x1, y1, x2, y2 = track.bbox

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        # build label — name + speed
        lines = []
        if self.show_name and stats and stats.name != "unknown":
            lines.append(stats.name)
        else:
            lines.append(f"id:{track.track_id}")

        if self.show_speed and stats and stats.speeds:
            speed_kmh = stats.speeds[-1] * 3.6
            lines.append(f"{speed_kmh:.1f} km/h")

        # draw label background + text above box
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness  = 1
        pad        = 3

        label_y = y1 - 4
        for i, line in enumerate(reversed(lines)):
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            ty = label_y - i * (th + pad + 2)
            cv2.rectangle(frame,
                          (x1, ty - th - pad),
                          (x1 + tw + pad * 2, ty + pad),
                          colour, -1)
            cv2.putText(frame, line,
                        (x1 + pad, ty),
                        font, font_scale, (255, 255, 255), thickness)

    def _draw_ball(self, frame, ball):
        x1, y1, x2, y2 = ball.bbox
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 8,  (0, 255, 255), -1)
        cv2.circle(frame, (cx, cy), 8,  (0, 0, 0),      1)
        cv2.putText(frame, "ball", (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    def _draw_minimap(self, frame, tracks, stats_map, ball) -> np.ndarray:
        """
        Draws a small top-down pitch diagram in the bottom-right corner
        showing every player's current pitch position.
        """
        s  = self.minimap_scale
        mw = int(self.pitch_length * s)
        mh = int(self.pitch_width  * s)
        pad = 8

        # blank green pitch
        minimap = np.full((mh, mw, 3), (60, 140, 60), dtype=np.uint8)

        # pitch lines
        cv2.rectangle(minimap, (0, 0), (mw - 1, mh - 1), (255, 255, 255), 1)
        cv2.line(minimap, (mw // 2, 0), (mw // 2, mh), (255, 255, 255), 1)
        cv2.circle(minimap, (mw // 2, mh // 2), int(9.15 * s), (255, 255, 255), 1)

        # players
        for track in tracks:
            stats = stats_map.get(track.track_id)
            if not stats or not stats.positions:
                continue
            _, x_m, y_m = stats.positions[-1]
            px = int(x_m * s)
            py = int(y_m * s)
            px = max(2, min(mw - 2, px))
            py = max(2, min(mh - 2, py))
            colour = self.team_colours.get(
                stats.team if stats else -1, (180, 180, 180)
            )
            cv2.circle(minimap, (px, py), 4, colour, -1)
            cv2.putText(minimap, str(track.track_id),
                        (px + 4, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)

        # ball
        if ball is not None:
            ball_pos = self.homography.pixel_to_pitch(
                (ball.bbox[0] + ball.bbox[2]) // 2,
                (ball.bbox[1] + ball.bbox[3]) // 2,
            )
            if ball_pos:
                bx = int(ball_pos.x_m * s)
                by = int(ball_pos.y_m * s)
                bx = max(2, min(mw - 2, bx))
                by = max(2, min(mh - 2, by))
                cv2.circle(minimap, (bx, by), 4, (0, 255, 255), -1)

        # place minimap in bottom-right corner of frame
        fh, fw = frame.shape[:2]
        x_off = fw - mw - pad
        y_off = fh - mh - pad

        # semi-transparent background
        roi = frame[y_off:y_off + mh, x_off:x_off + mw]
        cv2.addWeighted(minimap, 0.85, roi, 0.15, 0, roi)
        frame[y_off:y_off + mh, x_off:x_off + mw] = roi

        return frame

    def show(self, frame: np.ndarray, window_name: str = "Pipeline"):
        cv2.imshow(window_name, frame)
        return cv2.waitKey(1)

    def release(self):
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()